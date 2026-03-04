# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.chem import BOND_TYPES


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x
    

def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)   # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
    
    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added. 
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type
    

def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=0):

    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(
        edge_index, 
        edge_type, 
        torch.Size([N, N])
    )

    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index, 
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    new_edge_index = new_edge_index[:,batch[new_edge_index[0]]==batch[new_edge_index[1]]]
    new_edge_type = new_edge_type[batch[new_edge_index[0]]==batch[new_edge_index[1]]]
    
    return new_edge_index, new_edge_type


def extend_to_cross_attention(pos, cutoff, batch, graph_idx):

    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)
    rgraph_edge_index = rgraph_edge_index[:,graph_idx[rgraph_edge_index[0]]!=graph_idx[rgraph_edge_index[1]]]
    
    return rgraph_edge_index

def extend_to_cross_attention2(batch, graph_idx):

    rgraph_edge_index = torch.combinations(torch.arange(batch.size(0)).to(batch.device)).t()
    rgraph_edge_index = rgraph_edge_index[:,graph_idx[rgraph_edge_index[0]]!=graph_idx[rgraph_edge_index[1]]]
    rgraph_edge_index = rgraph_edge_index[:,batch[rgraph_edge_index[0]]==batch[rgraph_edge_index[1]]]
    
    return rgraph_edge_index

def create_topk_edges(node_embeddings, batch, graph_idx, k: int = 20):
    """
    배치 내 각 그래프에서 노드 임베딩을 기반으로 상위 K개의 이웃을 선택하여 엣지를 생성합니다.
    
    Args:
        node_embeddings (Tensor): 노드 임베딩, 크기 (N, D)
        batch (Tensor): 각 노드의 그래프 배치 정보, 크기 (N,)
        graph_idx (Tensor): 각 노드의 그래프 인덱스 정보, 크기 (N,) 
        k (int): 각 노드에 대해 연결할 이웃의 수
        
    Returns:
        edge_index (LongTensor): 엣지 인덱스, 크기 (2, E)
    """
    device = node_embeddings.device
    unique_graphs = batch.unique()
    edge_list = []

    for graph in unique_graphs:
        # 현재 그래프에 속한 노드 마스크
        mask = batch == graph
        
        # graph_idx에 따라 노드 분류
        node_emb0_mask = ((graph_idx % 2) == 0) & mask  # 짝수 graph_idx
        node_emb1_mask = ((graph_idx % 2) == 1) & mask  # 홀수 graph_idx

        node_emb0 = node_embeddings[node_emb0_mask]  # (N0, D)
        node_emb1 = node_embeddings[node_emb1_mask]  # (N1, D)

        N0, D0 = node_emb0.size()
        N1, D1 = node_emb1.size()

        if N0 == 0 or N1 == 0:
            continue  # 한 쪽 그룹에 노드가 없으면 엣지를 생성하지 않음

        # 코사인 유사도 계산: (N0, N1)
        similarity = F.cosine_similarity(
            node_emb0.unsqueeze(1).expand(-1, N1, -1),  # (N0, N1, D)
            node_emb1.unsqueeze(0).expand(N0, -1, -1),  # (N0, N1, D)
            dim=2
        )  # 결과: (N0, N1)

        # 자기 자신과의 유사도를 제외 (해당되지 않을 수 있음, 하지만 안전하게 -inf로 설정)
        # similarity.fill_diagonal_(-1e9)  # 이 경우 bipartite라 diagonal이 의미 없음

        # 상위 K개의 유사도 인덱스 선택: (N0, k)
        if k > N1:
            current_k = N1  # 선택할 수 있는 최대 K
        else:
            current_k = k
        topk_values, topk_indices = torch.topk(similarity, k=current_k, dim=1, largest=True, sorted=True)  # (N0, k)

        # 소스 노드 인덱스 (로컬 인덱스 -> 글로벌 인덱스)
        src_local = torch.arange(N0, device=device).unsqueeze(1).expand(-1, current_k).reshape(-1)  # (N0 * k,)
        tgt_local = topk_indices.reshape(-1)  # (N0 * k,)

        # 글로벌 인덱스로 변환
        src_global = torch.nonzero(node_emb0_mask).squeeze(1)[src_local]  # (N0 * k,)
        tgt_global = torch.nonzero(node_emb1_mask).squeeze(1)[tgt_local]  # (N0 * k,)

        # 엣지 추가
        edge_list.append(torch.stack([src_global, tgt_global], dim=0))  # (2, N0 * k)
        edge_list.append(torch.stack([tgt_global, src_global], dim=0))  # (2, N0 * k)

    if len(edge_list) > 0:
        edge_index = torch.cat(edge_list, dim=1)  # (2, E)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_index


def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch, order=3, cutoff=10.0, 
                              extend_order=True, extend_radius=True):
    
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, 
            edge_index=edge_index, 
            edge_type=edge_type, order=order
        )

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos, 
            edge_index=edge_index, 
            edge_type=edge_type, 
            cutoff=cutoff, 
            batch=batch,
        )
    
    return edge_index, edge_type