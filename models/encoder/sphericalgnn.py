import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from utils.geometry import *


class S_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, norm_diff=True, tanh=False, coords_range=1, norm_constant=0):
        super(S_GCL, self).__init__()
        input_edge = input_nf * 2
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.norm_constant = norm_constant
        spherical_coord_nf = 7


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + spherical_coord_nf + edges_in_d, hidden_nf),
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

    def edge_model(self, source, target, radial, coord_source, coord_target, edge_attr, edge_mask):
        spherical_coord_source = cartesian_to_spherical(coord_source)
        spherical_coord_target = cartesian_to_spherical(coord_target)
        # print(spherical_coord_source[:,0][None,...].shape)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, spherical_coord_source[:,0].unsqueeze(1), spherical_coord_target[:,0].unsqueeze(1), (spherical_coord_source[:,1:] - spherical_coord_target[:,1:]).cos(), (spherical_coord_source[:,1:] - spherical_coord_target[:,1:]).sin(), radial], dim=1)
            # out = torch.cat([source, target, spherical_coord_source[:,0].unsqueeze(1), spherical_coord_target[:,0].unsqueeze(1), radial], dim=1)
        else:
            out = torch.cat([source, target, spherical_coord_source[:,0].unsqueeze(1), spherical_coord_target[:,0].unsqueeze(1), (spherical_coord_source[:,1:] - spherical_coord_target[:,1:]).cos(), (spherical_coord_source[:,1:] - spherical_coord_target[:,1:]).sin(), radial, edge_attr], dim=1)
            # out = torch.cat([source, target, spherical_coord_source[:,0].unsqueeze(1), spherical_coord_target[:,0].unsqueeze(1), radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
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

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None, coord_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, coord[row], coord[col], edge_attr, edge_mask) # Make message
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr) # Update nodes

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + self.norm_constant)

        return radial, coord_diff


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
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", S_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, S_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, norm_constant=norm_constant))
        
        # self.dropout_h = nn.Dropout(0.05)

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None, coord_mask=None):
        # Edit Emiel: Remove velocity as input
        # edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        h = self.embedding(h)
        # h = self.dropout_h(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask, coord_mask=coord_mask)
            # h = self.dropout_h(h)
        h = self.embedding_out(h)
        # h = self.dropout_h(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result