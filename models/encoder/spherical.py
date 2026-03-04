import torch
from torch import nn
from utils.geometry import *

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


class S_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, norm_diff=True, tanh=False, coords_range=1, norm_constant=0):
        super(S_GCL, self).__init__()
        input_edge = input_nf * 2
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.norm_constant = norm_constant
        self.edges_in_d = edges_in_d

        # 중요: 이 값은 실제 cartesian_to_spherical 함수가 반환하는 특징의 차원 수와 일치해야 합니다.
        # 오류 분석 (실제 324 vs 기대 326)에 따르면 이 값은 3이어야 합니다.
        # 기존 코드에서는 5로 하드코딩 되어 있었음.
        actual_spherical_feature_dim = 3 

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + actual_spherical_feature_dim + 1 + self.edges_in_d, hidden_nf), # radial을 위해 +1 추가
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, coord_source, coord_target, edge_attr):
        # cartesian_to_spherical은 (N_edges, actual_spherical_feature_dim) 크기의 텐서를 반환해야 함
        spherical_coord_source = cartesian_to_spherical(coord_source)
        spherical_coord_target = cartesian_to_spherical(coord_target)
        
        # 이 차원은 actual_spherical_feature_dim과 일치해야 함 (여기서는 3으로 가정)
        spherical_coord_diff = spherical_coord_source - spherical_coord_target

        features_to_cat = [source, target, spherical_coord_diff, radial]

        if edge_attr is not None:
            if self.edges_in_d == 0:
                print("Warning: S_GCL initialized with edges_in_d=0 but edge_attr was provided. Ignoring edge_attr.")
            elif edge_attr.size(1) != self.edges_in_d:
                raise ValueError(f"Provided edge_attr has {edge_attr.size(1)} features, "
                                 f"but S_GCL was expecting {self.edges_in_d} features.")
            else:
                features_to_cat.append(edge_attr)
        else: 
            if self.edges_in_d > 0:
                placeholder_edge_attr = torch.zeros(source.size(0), self.edges_in_d, 
                                                    device=source.device, dtype=source.dtype)
                features_to_cat.append(placeholder_edge_attr)
        
        out = torch.cat(features_to_cat, dim=1)
        # 이제 out.shape[1]은 input_edge + actual_spherical_feature_dim + 1 + self.edges_in_d와 일치해야 함
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, coord[row], coord[col], edge_attr)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff_norm = coord_diff/(norm + self.norm_constant) 

        return radial, coord_diff_norm


class SphericalGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum', norm_constant=0, inv_sublayers=1, sin_embedding=False):
        super(SphericalGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean': 
            self.coords_range_layer = self.coords_range_layer * 19
        
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        for i in range(0, n_layers):
            # S_GCL 초기화 시 edges_in_d는 in_edge_nf로 전달됩니다.
            # actual_spherical_feature_dim은 S_GCL 내부에 설정된 값(여기서는 3)을 따릅니다.
            # 만약 이 값을 S_GCL 외부에서 제어하고 싶다면, S_GCL의 __init__ 파라미터로 추가해야 합니다.
            self.add_module(f"gcl_{i}", S_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                                              edges_in_d=in_edge_nf, 
                                              act_fn=act_fn, 
                                              attention=attention, 
                                              norm_diff=norm_diff, 
                                              tanh=tanh, 
                                              coords_range=self.coords_range_layer, 
                                              norm_constant=norm_constant))
        
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)

        return h



class SphericalGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum', norm_constant=0, inv_sublayers=1, sin_embedding=False):
        super(SphericalGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean': # This seems like a parameter for something else, not directly used in snippet
            self.coords_range_layer = self.coords_range_layer * 19
        
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        for i in range(0, n_layers):
            # When constructing S_GCL, pass in_edge_nf as edges_in_d
            # This tells S_GCL the expected dimension of edge_attr if provided.
            # If edge_attr is None in forward, S_GCL will handle it by padding if in_edge_nf > 0.
            self.add_module(f"gcl_{i}", S_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                                              edges_in_d=in_edge_nf, # Pass in_edge_nf here
                                              act_fn=act_fn, 
                                              attention=attention, 
                                              norm_diff=norm_diff, 
                                              tanh=tanh, 
                                              coords_range=self.coords_range_layer, 
                                              norm_constant=norm_constant))
        
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None):
        # The `edge_attr` argument to this forward method can be None.
        # If it's None, it will be passed as None to S_GCL layers.
        # S_GCL's edge_model will then handle this, potentially padding with zeros
        # if its self.edges_in_d (derived from in_edge_nf at initialization) is > 0.

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)

        return h