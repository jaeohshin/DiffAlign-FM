import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# 기존 unsorted_segment_sum 함수는 그대로 사용합니다.
def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

class NonEquivariant_GCL(nn.Module):
    """
    Non-equivariant Graph Convolutional Layer that uses coordinate information
    for message passing, but does not update coordinates equivariantly.
    It updates node features based on geometric relationships.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0,
                 act_fn=nn.SiLU(), attention=False, use_coord_diff=False, use_6d_rot_repr=True): # 6D 표현 사용 플래그 추가
        super(NonEquivariant_GCL, self).__init__()
        self.attention = attention
        self.use_coord_diff = use_coord_diff
        self.use_6d_rot_repr = use_6d_rot_repr

        input_edge = input_nf * 2  # source_h, target_h
        edge_coords_nf = 1  # For radial distance (squared)

        geom_dim = 0
        if self.use_6d_rot_repr:
            geom_dim = 6  # 6D rotation representation
        elif self.use_coord_diff:
            geom_dim = 3  # Normalized coordinate differences

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + geom_dim + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf), # Aggregated messages + original node features
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    @staticmethod
    def _get_6d_rot_from_direction(v_direction, eps=1e-8):
        """
        Generates a 6D rotation representation from a batch of 3D direction vectors.
        The first 3 dimensions are the normalized direction vector (v1).
        The next 3 dimensions are an orthogonal vector to v1 (v2).
        Args:
            v_direction (Tensor): Batch of 3D direction vectors (E, 3).
            eps (float): Small epsilon for numerical stability.
        Returns:
            Tensor: 6D rotation representation (E, 6).
        """
        # Normalize the primary direction vector (v1)
        v1_norm = torch.linalg.norm(v_direction, dim=-1, keepdim=True)
        v1 = v_direction / (v1_norm + eps) # (E, 3)

        # Create a second vector (v2) orthogonal to v1
        # A common method: choose a reference vector not collinear with v1.
        # If v1 is aligned with x-axis, use y-axis as ref. Otherwise, use x-axis.
        ref_x = torch.tensor([1.0, 0.0, 0.0], device=v1.device, dtype=v1.dtype).unsqueeze(0).expand_as(v1)
        ref_y = torch.tensor([0.0, 1.0, 0.0], device=v1.device, dtype=v1.dtype).unsqueeze(0).expand_as(v1)

        # Check alignment with x-axis
        dot_with_x = torch.sum(v1 * ref_x, dim=1, keepdim=True)
        is_aligned_with_x = torch.abs(dot_with_x) > (1.0 - eps)

        # Choose reference vector based on alignment
        reference_vector = torch.where(is_aligned_with_x, ref_y, ref_x)

        # Use Gram-Schmidt to find v2:
        # v2_unnormalized = reference_vector - <reference_vector, v1> * v1
        v2_unnormalized = reference_vector - torch.sum(reference_vector * v1, dim=1, keepdim=True) * v1
        v2_norm = torch.linalg.norm(v2_unnormalized, dim=-1, keepdim=True)
        v2 = v2_unnormalized / (v2_norm + eps) # (E, 3)

        return torch.cat([v1, v2], dim=-1) # (E, 6)

    def get_geometric_edge_features(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col] # (E, 3)
        radial_sq = torch.sum(coord_diff**2, 1).unsqueeze(1) # (E, 1)

        geom_features = None
        if self.use_6d_rot_repr:
            geom_features = self._get_6d_rot_from_direction(coord_diff) # (E, 6)
        elif self.use_coord_diff:
            # Normalize coord_diff (optional, but often helpful)
            # Add a small epsilon for numerical stability if normalizing
            norm = torch.sqrt(radial_sq + 1e-8) # norm is the actual radial distance
            geom_features = coord_diff / (norm + 1e-8) # (E, 3), Avoid division by zero
        
        return radial_sq, geom_features

    def edge_model(self, source_h, target_h, radial, geom_features, edge_attr, edge_mask):
        edge_input_list = [source_h, target_h, radial]
        if geom_features is not None: # This can be 3D coord_diff or 6D rot_repr
            edge_input_list.append(geom_features)
        if edge_attr is not None:
            edge_input_list.append(edge_attr)

        out = torch.cat(edge_input_list, dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, h, edge_index, edge_feat, node_attr):
        row, col = edge_index
        # Aggregate messages for each node
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))

        node_input_list = [h, agg]
        if node_attr is not None:
            node_input_list.append(node_attr)

        agg_cat = torch.cat(node_input_list, dim=1) # Renamed to avoid conflict
        # Perform MLP update, could be residual or direct
        out_h = self.node_mlp(agg_cat) # Update node features
        return out_h, agg # Return original aggregation for potential other uses

    def forward(self, h, coord, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index

        # Calculate geometric features from coordinates
        radial_sq, geom_features = self.get_geometric_edge_features(edge_index, coord)

        # Create edge messages (phi_e)
        edge_feat = self.edge_model(h[row], h[col], radial_sq, geom_features, edge_attr, edge_mask)

        # Update node features (phi_h)
        h_new, _ = self.node_model(h, edge_index, edge_feat, node_attr)

        if node_mask is not None:
            h_new = h_new * node_mask

        return h_new, coord, edge_attr


# 전체 MPNN 모델
class NonEquivariantGeometricGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(),
                 n_layers=4, attention=False, out_node_nf=None,
                 use_coord_diff_in_layers=False,
                 use_6d_rot_repr_in_layers=True # 6D 표현 사용 플래그 추가
                ):
        super(NonEquivariantGeometricGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f"gcl_{i}",
                            NonEquivariant_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                               edges_in_d=in_edge_nf,
                                               act_fn=act_fn,
                                               attention=attention,
                                               use_coord_diff=use_coord_diff_in_layers,
                                               use_6d_rot_repr=use_6d_rot_repr_in_layers # 플래그 전달
                                              ))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding_in(h)

        for i in range(0, self.n_layers):
            gcl_layer = self._modules[f"gcl_{i}"]
            h, _, _ = gcl_layer(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)

        h = self.embedding_out(h)

        if node_mask is not None:
            h = h * node_mask
        return h