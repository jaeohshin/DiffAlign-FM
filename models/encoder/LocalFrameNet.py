from torch import nn
import torch
from itertools import combinations # 각도 계산을 위해 추가

# (기존 unsorted_segment_sum, unsorted_segment_mean, get_edges, get_edges_batch 함수는 동일하다고 가정)
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    # edge_attr = torch.ones(len(edges[0]) * batch_size, 1) # Removed edge_attr generation here
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        # return edges, edge_attr
        return edges # Return only edges
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    # return edges, edge_attr
    return edges # Return only edges

# --- 수정된 L_GCL 클래스 ---
class L_GCL_Angle(nn.Module):
    """
    E(n) Equivariant Convolutional Layer - Modified to include angle information.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, angle_nf=16):
        super(L_GCL_Angle, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg # Note: coord aggregation logic not fully shown/used in original forward
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.angle_nf = angle_nf # Dimension for angle features

        # MLP for edge features based on pairwise interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # MLP to process angle information (cosine values)
        if self.angle_nf > 0:
            self.angle_mlp = nn.Sequential(
                nn.Linear(1, self.angle_nf), # Input is cosine value
                act_fn,
                nn.Linear(self.angle_nf, self.angle_nf),
                act_fn
            )
            # Adjust node_mlp input size to include aggregated angle features
            node_mlp_input_dim = hidden_nf + input_nf + self.angle_nf
        else:
            self.angle_mlp = None
            node_mlp_input_dim = hidden_nf + input_nf

        # MLP for node feature updates
        self.node_mlp = nn.Sequential(
            nn.Linear(node_mlp_input_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # Optional coordinate update MLP (structure kept from original, but not used in fwd return)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # Optional attention mechanism for edges
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def _get_neighbors(self, edge_index, num_nodes):
        """Helper to quickly get neighbors for each node."""
        adj = [[] for _ in range(num_nodes)]
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj[i].append(j)
        return adj

    def _calculate_angles(self, coord, edge_index):
        """Calculates cosine of angles i-j-k for all nodes j."""
        num_nodes = coord.size(0)
        row, col = edge_index
        adj = self._get_neighbors(edge_index, num_nodes)

        # Stores angle cosines, indexed by the central node j
        angle_cosines = [[] for _ in range(num_nodes)]
        # Stores corresponding neighbor pairs (i, k) for each angle at j
        angle_neighbor_pairs = [[] for _ in range(num_nodes)]

        for j in range(num_nodes):
            neighbors = adj[j]
            # Need at least two neighbors to form an angle
            if len(neighbors) >= 2:
                # Iterate through all unique pairs of neighbors (i, k) for node j
                for i, k in combinations(neighbors, 2):
                    # Vectors from center node j to neighbors i and k
                    v_ji = coord[i] - coord[j]
                    v_jk = coord[k] - coord[j]

                    # Normalize vectors
                    norm_ji = torch.linalg.norm(v_ji, dim=-1) + self.epsilon
                    norm_jk = torch.linalg.norm(v_jk, dim=-1) + self.epsilon

                    # Calculate cosine: dot(v_ji, v_jk) / (||v_ji|| * ||v_jk||)
                    cos_angle = torch.sum(v_ji * v_jk, dim=-1) / (norm_ji * norm_jk)

                    # Clamp for numerical stability (acos domain is [-1, 1])
                    cos_angle = torch.clamp(cos_angle, -1.0 + self.epsilon, 1.0 - self.epsilon)

                    angle_cosines[j].append(cos_angle)
                    angle_neighbor_pairs[j].append((i,k)) # Store which neighbors formed this angle

        # Convert lists to tensors where possible, handle nodes with < 2 neighbors
        # We need a way to aggregate these per node for the node_model
        # Let's process them within the forward pass directly for simplicity here

        return angle_cosines, angle_neighbor_pairs # Return raw cosines and pairs

    def edge_model(self, source, target, radial, edge_attr):
        """Computes edge features."""
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_feat, angle_features, node_attr):
        """Updates node features using aggregated edge and angle features."""
        row, col = edge_index
        # Aggregate edge features (standard message passing)
        agg_edge = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))

        # Prepare input for node MLP
        if node_attr is not None:
            node_input = torch.cat([x, agg_edge, angle_features, node_attr], dim=1)
        else:
            node_input = torch.cat([x, agg_edge, angle_features], dim=1) # Concatenate angle features

        # Apply node MLP
        out = self.node_mlp(node_input)

        # Apply residual connection
        if self.residual:
            out = x + out
        return out # Return only updated node features 'h'

    def coord2radial(self, edge_index, coord):
        """Calculates squared radial distances and optionally normalized coord differences."""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm # Note: coord_diff isn't used later in this version

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        num_nodes = h.size(0)
        row, col = edge_index

        # 1. Calculate radial features (distances) for edges
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # 2. Calculate edge features using edge_model
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # 3. Calculate Angle Features (if angle_nf > 0)
        if self.angle_nf > 0 and self.angle_mlp is not None:
            # Calculate raw angle cosines for i-j-k centered at j
            angle_cosines_list, _ = self._calculate_angles(coord, edge_index)

            # Process and aggregate angle features for each node
            aggregated_angle_features = h.new_zeros((num_nodes, self.angle_nf))
            for j in range(num_nodes):
                if angle_cosines_list[j]: # Check if node j has any angles
                    # Stack cosines for node j into a tensor
                    cosines_j = torch.stack(angle_cosines_list[j]).unsqueeze(-1) # Shape: [num_angles_j, 1]
                    # Pass through angle MLP
                    angle_feats_j = self.angle_mlp(cosines_j) # Shape: [num_angles_j, angle_nf]
                    # Aggregate angle features for node j (e.g., mean or sum)
                    aggregated_angle_features[j] = torch.mean(angle_feats_j, dim=0)
        else:
            # If not using angles, create a zero tensor placeholder
            aggregated_angle_features = h.new_zeros((num_nodes, 0)) # Shape: [N, 0] if angle_nf=0

        # 4. Update node features using node_model (now includes angle features)
        # Note: The original node_model signature included 'agg' as output, removed here for clarity
        h_new = self.node_model(h, edge_index, edge_feat, aggregated_angle_features, node_attr)

        # 5. Coordinate update (optional, based on original structure but not returned)
        # coord_update = self.coord_model(h_new, edge_index, coord_diff, edge_feat)
        # coord = coord + coord_update # Example, logic depends on coord_model implementation

        # Return updated node features h. Coordinates are not updated/returned in this version.
        return h_new, coord, edge_attr # Return updated h, original coord, original edge_attr


# --- 수정된 LocalFrameNet 클래스 (L_GCL_Angle 사용) ---
class LocalFrameNet(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, angle_nf=16): # Added angle_nf
        super(LocalFrameNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            # Use the modified L_GCL_Angle layer
            self.add_module("gcl_%d" % i, L_GCL_Angle(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                      edges_in_d=in_edge_nf,
                                                      act_fn=act_fn, residual=residual, attention=attention,
                                                      normalize=normalize, tanh=tanh, angle_nf=angle_nf)) # Pass angle_nf
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            # Call the modified layer
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x # Return final node features and potentially unchanged coordinates