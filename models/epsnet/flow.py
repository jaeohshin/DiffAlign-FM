# ===== Standard library =====
import math
from typing import Optional, Tuple

# ===== Third-party =====
import torch
from torch import autograd
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn as _knn
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_max
from uff_torch import UFFTorch, build_uff_inputs, merge_uff_inputs
from vinasf_torch import VinaSFTorch

# ===== Local (project) =====
# NOTE: 기존 프로젝트 구조를 유지합니다. 아래 경로는 원본 DiffAlign이 쓰던 것과 동일.
from ..encoder.egnn import EGNN
from ..encoder.edge import MLPEdgeEncoder
from ..common import extend_graph_order_radius

from concurrent.futures import ProcessPoolExecutor   # [ADD]
import multiprocessing as mp     
from rdkit import Chem                                # [ADD]
from rdkit.Chem import AllChem                        # [ADD]
from rdkit.Geometry import Point3D                    # [ADD]
import numpy as np   

# ===== UFF helper (single-mol gradient worker) =====
def _uff_grad_worker2(args):
    """RDKit UFF: coords → (-∇E) 반환. args = (q_bytes, _, coords)"""
    q_bytes, _unused, coords = args
    qm = Chem.Mol(q_bytes) if isinstance(q_bytes, bytes) else Chem.Mol(q_bytes.ToBinary())
    conf = qm.GetConformer(0)
    for a, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(int(a), Point3D(float(x), float(y), float(z)))
    ff = AllChem.UFFGetMoleculeForceField(qm)
    grad = ff.CalcGrad()
    g = np.asarray(grad, dtype=np.float32).reshape(-1, 3)
    return -g  # (-∇E)

# ---------------- Schedules ----------------

def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal (2021): https://arxiv.org/abs/2102.09672
    Returns betas of length T (float32).
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


# ---------------- Positional / time encoders ----------------

class SinusoidalPosEmb(nn.Module):
    """타임스텝용 사인·코사인 임베딩 (float32)."""
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension ({dim}) must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        t = t.float()
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        pos = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((pos.sin(), pos.cos()), dim=-1).float()


class DDPMTimeEncoder(nn.Module):
    """SinusoidalPosEmb + MLP로 타임스텝 임베딩."""
    def __init__(self, embed_dim: int, activation=nn.SiLU):
        super().__init__()
        sine_embed_dim = embed_dim if (embed_dim % 2 == 0) else (embed_dim - 1)
        self.pos_emb = SinusoidalPosEmb(sine_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sine_embed_dim, embed_dim),
            activation(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_emb(t)).float()


# ---------------- Geometry helpers ----------------

def get_distance(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=pos.dtype, device=pos.device)
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


# ---------------- Cross-graph attention (same spirit as original) ----------------


class CrossAttention(nn.Module):
    """
    Q <- R masked dense cross-attention.

    v1 패치:
      - null token 쪽으로 간 확률을 atom-wise gate로 사용해서
        좌표 업데이트 강도를 조절함.
      - feature(h)는 기존과 완전히 동일하게 업데이트.
      - coord_update=True일 때만 좌표 업데이트 수행.
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, coord_update: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.coord_update = coord_update

        # 프로젝션/정규화/FF
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.ff_q = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        # Null token (헤드별)
        self.k_null = nn.Parameter(torch.randn(self.heads, self.dh) * 0.02)   # [H,Dh]
        self.v_null = nn.Parameter(torch.randn(self.heads, self.dh) * 0.02)   # [H,Dh]
        self.b_null = nn.Parameter(torch.full((self.heads,), -1.0))           # [H]

        # 좌표 업데이트 (EGNN-like)
        self.coord_edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.coord_scalar = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_uniform_(self.coord_scalar.weight, gain=1e-2)

    @staticmethod
    def _masked_logsumexp(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        # logits: [..., L], mask: [..., L] (bool)
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
        m = torch.amax(masked, dim=dim, keepdim=True)
        sumexp = torch.sum(torch.exp(masked - m), dim=dim, keepdim=True)
        return (m + torch.log(sumexp + eps)).squeeze(dim)

    # --- 레이어 시작 시 R측 LN/Proj 1회 준비 ---
    def preproject_R(self, h_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h_r: [B,M,D]  ->  (rh [B,M,D], k_r [B,H,M,Dh], v_r [B,H,M,Dh])
        """
        rh = self.norm_r(h_r)
        k_lin = self.k_proj(rh).contiguous().view(
            rh.size(0), rh.size(1), self.heads, self.dh
        )  # [B,M,H,Dh]
        v_lin = self.v_proj(h_r).contiguous().view(
            h_r.size(0), h_r.size(1), self.heads, self.dh
        )  # [B,M,H,Dh]
        k_r = k_lin.permute(0, 2, 1, 3).contiguous()  # [B,H,M,Dh]
        v_r = v_lin.permute(0, 2, 1, 3).contiguous()  # [B,H,M,Dh]
        return rh, k_r, v_r

    # --- 마스킹된 dense path (메인) ---
    def forward_dense(
        self,
        h_q: torch.Tensor,
        x_q: torch.Tensor,
        h_r: torch.Tensor,
        x_r: torch.Tensor,
        mask_q: torch.Tensor,
        mask_r: torch.Tensor,
        *,
        pre_k: torch.Tensor | None = None,
        pre_v: torch.Tensor | None = None,
        pre_rh: torch.Tensor | None = None,
        coord_chunk_M: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_q: [B,N,D], x_q: [B,N,3], mask_q: [B,N] (True=valid)
        h_r: [B,M,D], x_r: [B,M,3], mask_r: [B,M]
        """
        B, N, D = h_q.shape
        _, M, _ = h_r.shape
        H, Dh = self.heads, self.dh
        inv_sqrt_dh = 1.0 / math.sqrt(Dh)

        # Query 정규화/프로젝션
        qh = self.norm_q(h_q)                                      # [B,N,D]
        q_lin = self.q_proj(qh).contiguous().view(B, N, H, Dh)     # [B,N,H,Dh]
        q = q_lin.permute(0, 2, 1, 3).contiguous()                 # [B,H,N,Dh]

        # Reference 프로젝션(레이어당 1회)
        if pre_k is None or pre_v is None or pre_rh is None:
            rh, k_r, v_r = self.preproject_R(h_r)
        else:
            rh, k_r, v_r = pre_rh, pre_k, pre_v                    # [B,M,D], [B,H,M,Dh], [B,H,M,Dh]

        # 마스크 브로드캐스트
        mq = mask_q[:, None, :, None]       # [B,1,N,1]
        mr = mask_r[:, None, None, :]       # [B,1,1,M]
        pair_mask = mq & mr                  # [B,1,N,M]

        # logits (real)
        logits_real = torch.einsum(
            "bhnd,bhdm->bhnm",
            q,
            k_r.transpose(-1, -2),
        ) * inv_sqrt_dh  # [B,H,N,M]
        neg_inf = torch.finfo(logits_real.dtype).min
        logits_real = torch.where(
            pair_mask,
            logits_real,
            torch.full_like(logits_real, neg_inf),
        )

        # logits (null)
        logits_null = torch.einsum("bhnd,hd->bhn", q, self.k_null) * inv_sqrt_dh  # [B,H,N]
        logits_null = logits_null + self.b_null.view(1, H, 1)

        # LogSumExp (real + null)
        lse_real = self._masked_logsumexp(logits_real, pair_mask, dim=-1)     # [B,H,N]
        lse_all = torch.logaddexp(lse_real, logits_null)                      # [B,H,N]

        # attention weights
        alpha_real = torch.exp(logits_real - lse_all[:, :, :, None])          # [B,H,N,M]
        alpha_null = torch.exp(logits_null - lse_all)                          # [B,H,N]
        alpha_real = self.dropout(alpha_real)
        alpha_null = self.dropout(alpha_null)

        # -------- feature 업데이트 --------
        msg_real = torch.einsum("bhnm,bhmd->bhnd", alpha_real, v_r)           # [B,H,N,Dh]
        msg_null = alpha_null[:, :, :, None] * self.v_null.view(1, H, 1, Dh)  # [B,H,N,Dh]
        h_msg = (msg_real + msg_null).permute(0, 2, 1, 3).contiguous().view(B, N, D)  # [B,N,D]

        h_out = h_q + self.out_proj(h_msg)
        h_out = h_out + self.ff_q(h_out)

        # -------- 좌표 업데이트 없음인 경우 --------
        if (not self.coord_update) or M == 0 or N == 0:
            return h_out, x_q

        # ======================================================
        # v1: atom-wise coord gating with null probability
        # ------------------------------------------------------
        # alpha_real: [B,H,N,M], alpha_null: [B,H,N]
        # p_real(q) = Σ_m alpha_real(q,m)
        # gate_q ∈ [gate_min, 1] 로 좌표 업데이트 강도 조절
        # ======================================================

        # head별 p_real: [B,H,N]
        p_real = alpha_real.sum(dim=-1).clamp(0.0, 1.0)
        # head 평균: [B,N]
        gate_q = p_real.mean(dim=1)
        gate_q = gate_q.unsqueeze(-1)                  # [B,N,1]

        # -------- 좌표 업데이트 (기존 EGNN-like) --------
        alpha_s = alpha_real.mean(dim=1)  # [B,N,M]
        qh_new = self.norm_q(h_out)       # [B,N,D]
        rh_use = self.norm_r(h_r) if pre_rh is None else pre_rh  # [B,M,D]

        dx = torch.zeros_like(x_q)        # [B,N,3]
        chunk = max(1, int(coord_chunk_M))
        for m0 in range(0, M, chunk):
            m1 = min(M, m0 + chunk)
            mr_chunk = mask_r[:, m0:m1]               # [B,m]
            if not torch.any(mr_chunk):
                continue

            # [B,N,m,3]
            diff = x_q[:, :, None, :] - x_r[:, None, m0:m1, :]
            radial = (diff * diff).sum(dim=-1, keepdim=True)          # [B,N,m,1]
            dirn = diff * torch.rsqrt(radial + 1e-8)                  # [B,N,m,3]

            # features: [B,N,m,2D+1]
            qh_blk = qh_new[:, :, None, :].expand(-1, -1, m1 - m0, -1)
            rh_blk = rh_use[:, None, m0:m1, :].expand(-1, N, -1, -1)
            edge_in = torch.cat([qh_blk, rh_blk, radial], dim=-1)
            s = self.coord_scalar(self.coord_edge_mlp(edge_in)).squeeze(-1)  # [B,N,m]

            valid = (mask_q[:, :, None] & mr_chunk[:, None, :])              # [B,N,m]
            w = torch.where(
                valid,
                alpha_s[:, :, m0:m1],
                torch.zeros_like(alpha_s[:, :, m0:m1]),
            )
            s = torch.where(valid, s, torch.zeros_like(s))

            dx += (dirn * (s * w).unsqueeze(-1)).sum(dim=2)

        # ---- atom-wise gate 적용 ----
        dx = dx * gate_q  # [B,N,3]

        x_out = x_q + dx
        return h_out, x_out

    # --- 기존 시그니처 유지 (CrossGraphAligner에서 호출) ---
    def forward(self, h_q, x_q, h_r, x_r, q2r_edge_index=None, k_train: int = None):
        """
        기존 코드와의 호환을 위해 남겨둔 인터페이스.
        - q2r_edge_index는 무시하고 dense 경로로 수행.
        - 배치는 CrossGraphAligner에서 패킹해 넘겨준다.
        """
        # 패딩된 토큰들은 0으로 채웠다는 가정하에 isfinite로 마스크 생성
        mask_q = torch.any(torch.isfinite(h_q), dim=-1)  # [B,N]
        mask_r = torch.any(torch.isfinite(h_r), dim=-1)  # [B,M]
        return self.forward_dense(h_q, x_q, h_r, x_r, mask_q, mask_r)


class CrossGraphAligner(nn.Module):
    """
    Masked dense cross-attn용 알라이너 (drop-in).
    - 시그니처/필드 이름 유지
    - 내부에서 (Q,R) 페어별로 패킹(N_max, M_max), 레이어마다 R측 pre-proj 1회
    - recompute_each 파라미터 유지(현재 dense에서는 효과 없음; 호환성용)
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0,
                 coord_update: bool = True,
                 num_layers: int = 6, recompute_each: int = 1,
                 coord_chunk_M: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttention(dim=dim, heads=heads, dropout=dropout, coord_update=coord_update)
            for _ in range(num_layers)
        ])
        self.recompute_each = int(recompute_each)
        self.coord_chunk_M = int(coord_chunk_M)

    # (호환성 유지용; 사용하지 않음)
    @staticmethod
    @torch.no_grad()
    def knn_q2r_edges(*args, **kwargs):
        return torch.empty((2, 0), dtype=torch.long, device=kwargs.get('x_q', torch.tensor((), device='cpu')).device if isinstance(kwargs, dict) and 'x_q' in kwargs else 'cpu')

    @staticmethod
    def _pack_by_pair(h, x, batch, graph_idx):
        """
        merged 배치에서 (짝수=Q, 홀수=R) 기준으로 페어별 Q/R를 추출하여
        [B,N,D], [B,M,D]로 패딩하고, 마스크/인덱스 테이블을 만든다.
        """
        device = h.device
        qmask = (graph_idx % 2 == 0)
        rmask = ~qmask
        idx_q_all = torch.nonzero(qmask, as_tuple=False).view(-1)
        idx_r_all = torch.nonzero(rmask, as_tuple=False).view(-1)
        if idx_q_all.numel() == 0 or idx_r_all.numel() == 0:
            return None

        pair_ids = torch.unique(batch)
        B = pair_ids.numel()

        q_lists, r_lists = [], []
        maxN = 0; maxM = 0
        for pid in pair_ids:
            q_idx = idx_q_all[(batch[idx_q_all] == pid)]
            r_idx = idx_r_all[(batch[idx_r_all] == pid)]
            q_lists.append(q_idx); r_lists.append(r_idx)
            maxN = max(maxN, q_idx.numel())
            maxM = max(maxM, r_idx.numel())

        D = h.size(-1)
        # 패딩 버퍼 (0으로 채움; 마스크로 유효토큰 구분)
        h_q = torch.zeros((B, maxN, D), device=device, dtype=h.dtype)
        x_q = torch.zeros((B, maxN, 3), device=device, dtype=x.dtype)
        h_r = torch.zeros((B, maxM, D), device=device, dtype=h.dtype)
        x_r = torch.zeros((B, maxM, 3), device=device, dtype=x.dtype)
        mask_q = torch.zeros((B, maxN), device=device, dtype=torch.bool)
        mask_r = torch.zeros((B, maxM), device=device, dtype=torch.bool)

        for b, (q_idx, r_idx) in enumerate(zip(q_lists, r_lists)):
            n, m = q_idx.numel(), r_idx.numel()
            if n > 0:
                h_q[b, :n] = h[q_idx]
                x_q[b, :n] = x[q_idx]
                mask_q[b, :n] = True
            if m > 0:
                h_r[b, :m] = h[r_idx]
                x_r[b, :m] = x[r_idx]
                mask_r[b, :m] = True

        return {
            "pair_ids": pair_ids,
            "q_lists": q_lists, "r_lists": r_lists,
            "h_q": h_q, "x_q": x_q, "mask_q": mask_q,
            "h_r": h_r, "x_r": x_r, "mask_r": mask_r,
        }

    @staticmethod
    def _unpack_Q(h_q_new, x_q_new, pack, h_global, x_global):
        """packed Q 결과를 글로벌 텐서에 되써넣기."""
        for b, q_idx in enumerate(pack["q_lists"]):
            n = q_idx.numel()
            if n == 0: 
                continue
            h_global[q_idx] = h_q_new[b, :n]
            x_global[q_idx] = x_q_new[b, :n]
        return h_global, x_global

    def forward(self, h, x, batch, graph_idx):
        pack = self._pack_by_pair(h, x, batch, graph_idx)
        if pack is None:
            return h, x

        h_q = pack["h_q"]; x_q = pack["x_q"]; mask_q = pack["mask_q"]
        h_r = pack["h_r"]; x_r = pack["x_r"]; mask_r = pack["mask_r"]

        # 레이어별 R측 pre-proj 1회
        pre_list = []
        for layer in self.layers:
            rh, k_r, v_r = layer.preproject_R(h_r)
            pre_list.append((rh, k_r, v_r))

        # 레이어 스택 (dense path)
        for li, layer in enumerate(self.layers):
            rh, k_r, v_r = pre_list[li]
            h_q, x_q = layer.forward_dense(
                h_q, x_q, h_r, x_r, mask_q, mask_r,
                pre_k=k_r, pre_v=v_r, pre_rh=rh,
                coord_chunk_M=self.coord_chunk_M,
            )

        # 글로벌 Q로 언팩
        h, x = self._unpack_Q(h_q, x_q, pack, h, x)
        return h, x


# ---------------- Batch merge ----------------

def _pick_device(*objs) -> torch.device:
    """
    Pick a common device from a list of tensors/Batches.
    Priority: first CUDA device encountered; else CPU.
    """
    for o in objs:
        if isinstance(o, torch.Tensor):
            if o.is_cuda:
                return o.device
        elif isinstance(o, Batch):
            # Try representative attribute
            for attr in ("pos", "x", "atom_type", "edge_index"):
                if hasattr(o, attr) and getattr(o, attr) is not None:
                    t = getattr(o, attr)
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        return t.device
    return torch.device("cpu")

def merge_graphs_in_batch(batch1: Batch, batch2: Batch, device: Optional[torch.device] = None) -> Batch:
    """
    Merge as [Q1,R1,Q2,R2,...] and attach graph_idx (even=query, odd=ref) and pair-level batch.
    All Data objects are moved to `device` beforehand to avoid CPU/CUDA mixing.
    """
    if device is None:
        device = _pick_device(batch1, batch2)

    data_list = []
    for d1, d2 in zip(batch1.to_data_list(), batch2.to_data_list()):
        data_list.append(d1.to(device))
        data_list.append(d2.to(device))
    if not data_list:
        # empty Batch on target device
        empty = Batch()
        # attach empty required attrs on correct device if needed later
        return empty

    merge_batch = Batch.from_data_list(data_list)  # now all on same device
    num_nodes_list = [d.num_nodes for d in data_list]

    # graph_idx/batch on correct device
    graph_idx_list = [torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(num_nodes_list)]
    batch_idx_list = [torch.full((n,), i // 2, dtype=torch.long, device=device) for i, n in enumerate(num_nodes_list)]

    merge_batch.graph_idx = torch.cat(graph_idx_list) if graph_idx_list else torch.empty(0, dtype=torch.long, device=device)
    merge_batch.batch = torch.cat(batch_idx_list) if batch_idx_list else torch.empty(0, dtype=torch.long, device=device)
    return merge_batch


# ---------------- Main (Isotropic DiffAlign) ----------------

class FlowAlign(nn.Module):
    """
    Flow Matching (conditional, isotropic)
    - 백본: EGNN + CrossGraphAligner (Query만 좌표 업데이트)
    - 출력: v_t (merged(Q,R) 순서)
    - 학습: v MSE + x0 anchor + repulsion(옵션)
    """
    def __init__(
        self,
        node_feature_dim: int = 64,
        time_embed_dim: int = 32,
        query_embed_dim: int = 32,
        edge_encoder_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gnn_layers_intra: int = 12,
        gnn_layers_intra_2: int = 4,
        gnn_layers_inter: int = 8,
        max_atom_types: int = 100,

        # Flow matching
        num_steps: int = 100,  # ODE steps at inference
        beta_start: float = 1e-4,  # unused, kept for compat
        beta_end: float = 0.02,    # unused, kept for compat
        schedule_type: str = 'cosine',  # unused, kept for compat

        # Repulsion
        repulsion_weight: float = 1e-2,
        repulsion_margin: float = 1.2,
        repulsion_exclude_hops: int = 3,
    ):
        super().__init__()

        # ---- Flow matching (no schedule buffers needed) ----
        self.num_steps = int(num_steps)

        # ---- Encoders ----
        self.edge_encoder = MLPEdgeEncoder(edge_encoder_dim, "relu")
        self.edge_encoder2 = MLPEdgeEncoder(edge_encoder_dim, "relu")

        self.node_encoder = nn.Sequential(
            nn.Embedding(max_atom_types, node_feature_dim),
            nn.SiLU(),
            nn.Linear(node_feature_dim, node_feature_dim),
        )
        self.time_encoder = DDPMTimeEncoder(time_embed_dim, activation=nn.SiLU)
        self.query_encoder = nn.Sequential(
            nn.Embedding(2, query_embed_dim),  # 0=ref, 1=query
            nn.SiLU(),
            nn.Linear(query_embed_dim, query_embed_dim),
        )

        gnn_in_node_dim = node_feature_dim + time_embed_dim + query_embed_dim

        self.intra_encoder = EGNN(
            in_node_nf=gnn_in_node_dim, in_edge_nf=edge_encoder_dim, hidden_nf=gnn_hidden_dim,
            n_layers=gnn_layers_intra, attention=True
        )
        self.cross_aligner = CrossGraphAligner(
            dim=gnn_hidden_dim,
            heads=4,
            dropout=0.1,
            coord_update=True,
            num_layers=gnn_layers_inter,
            recompute_each=1,
        )
        self.intra_encoder_2 = EGNN(
            in_node_nf=gnn_hidden_dim, in_edge_nf=edge_encoder_dim, hidden_nf=gnn_hidden_dim,
            n_layers=gnn_layers_intra_2, attention=True
        )

        # Repulsion hyperparams
        self.repulsion_weight = float(repulsion_weight)
        self.repulsion_margin = float(repulsion_margin)
        self.repulsion_exclude_hops = int(repulsion_exclude_hops)

    # -------------- Forward: predict v_t --------------

    def forward(self, query_batch: Batch, reference_batch: Batch, t: torch.Tensor,
                condition: bool = True) -> torch.Tensor:
        """
        입력:
          - query_batch, reference_batch: torch_geometric.data.Batch
          - t: [G] (그래프별 타임스텝, 0..T-1)
        출력:
          - v_hat (merged(Q,R) 순서의 [N_total, 3])
        """
        merged_batch = merge_graphs_in_batch(query_batch, reference_batch)
        if merged_batch.num_nodes == 0:
            device_to_use = query_batch.pos.device if hasattr(query_batch, 'pos') else 'cpu'
            return torch.zeros((0, 3), device=device_to_use)

        device = merged_batch.pos.device
        x_in = merged_batch.pos

        # (A) Query mask: 좌표 업데이트는 Q만
        qmask_bool = ((merged_batch.graph_idx % 2) == 0)
        coord_mask = qmask_bool.float().unsqueeze(-1)

        # (B) 임베딩
        node_feat = self.node_encoder(merged_batch.atom_type)
        t_nodes = t[merged_batch.batch]  # 그래프별 t를 노드로 확장
        time_emb = self.time_encoder(t_nodes)
        is_query = ((merged_batch.graph_idx % 2) == 0).long()  # 1=query, 0=ref
        query_emb = self.query_encoder(is_query)
        h = torch.cat([node_feat, time_emb, query_emb], dim=-1)

        # (C) Intra-graph (stage 1): Q만 좌표 업데이트
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=x_in,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=8,
            extend_order=True,
            extend_radius=True,
        )
        edge_length = get_distance(x_in, edge_index).unsqueeze(-1)
        e = self.edge_encoder(edge_length=edge_length, edge_type=edge_type)

        h, x = self.intra_encoder(
            h=h, x=x_in, edges=edge_index, edge_attr=e,
            coord_mask=coord_mask,
        )

        # (D) Cross (Q만 이동)
        if condition:
            h, x = self.cross_aligner(h, x, batch=merged_batch.batch, graph_idx=merged_batch.graph_idx)

        # (E) Intra-graph (stage 2)
        edge_index2, edge_type2 = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=x,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=8,
            extend_order=True,
            extend_radius=True,
        )
        edge_length2 = get_distance(x, edge_index2).unsqueeze(-1)
        e2 = self.edge_encoder2(edge_length=edge_length2, edge_type=edge_type2)

        h, x = self.intra_encoder_2(
            h=h, x=x, edges=edge_index2, edge_attr=e2,
            coord_mask=coord_mask,
        )

        # (F) 출력: v = x - x_in
        v_hat = x - x_in
        return v_hat

    # -------------- Training loss --------------

    def get_loss(self, query_batch: Batch, reference_batch: Batch, clamp: float = 1e-10):
        """
        Flow matching loss:
        - t ~ U[0,1] per graph
        - x_t = (1-t)*eps + t*x0  (linear interpolation)
        - v_target = x0 - eps      (constant velocity)
        - loss = MSE(v_pred, v_target) + x0_anchor + repulsion
        """
        if query_batch.num_nodes == 0:
            device = reference_batch.pos.device if hasattr(reference_batch, 'pos') and reference_batch.num_nodes > 0 else 'cpu'
            return torch.tensor(0.0, device=device, requires_grad=True)

        device = query_batch.pos.device
        x0 = query_batch.pos
        num_graphs = query_batch.num_graphs

        # t ~ U[0,1] per graph (continuous)
        t_graph = torch.rand(num_graphs, device=device)                         # [G]

        # broadcast t to nodes
        t_n = t_graph[query_batch.batch].unsqueeze(1)                           # [Nq,1]

        # linear interpolation: x_t = (1-t)*eps + t*x0
        eps = torch.randn_like(x0)
        x_t = (1.0 - t_n) * eps + t_n * x0                                     # [Nq,3]

        # constant velocity target
        v_target = x0 - eps                                                      # [Nq,3]

        noisy_query = query_batch.clone(); noisy_query.pos = x_t

        # velocity prediction (CFG dropout 20%)
        if self.training and (torch.rand(1, device=device) < 0.2):
            v_pred_merged = self(noisy_query, reference_batch, t_graph, condition=False)
        else:
            v_pred_merged = self(noisy_query, reference_batch, t_graph, condition=True)

        merged = merge_graphs_in_batch(noisy_query, reference_batch)
        qmask = (merged.graph_idx % 2 == 0)
        v_pred = v_pred_merged[qmask]                                           # [Nq,3]

        # velocity loss
        v_loss = F.mse_loss(v_pred, v_target, reduction='mean')

        # x0 anchor: x0_hat = x_t + (1-t)*v_pred
        x0_hat = x_t + (1.0 - t_n) * v_pred
        x0_loss = F.mse_loss(x0_hat, x0, reduction='mean')

        # repulsion
        rep_loss = self._repulsion_loss(query_batch, x0_hat)

        w_v   = float(getattr(self, 'v_loss_weight',   1.0))
        w_x0  = float(getattr(self, 'x0_loss_weight',  1.0))
        w_rep = float(getattr(self, 'repulsion_weight', self.repulsion_weight))
        loss = w_v*v_loss + w_x0*x0_loss + w_rep*rep_loss
        return loss
    
    @torch.no_grad() # Added
    def _repulsion_loss(self, qb: Batch, x0_hat: torch.Tensor) -> torch.Tensor:
        """k-hop(≤repulsion_exclude_hops) 쌍 제외 후 마진 기반 충돌 페널티."""
        device = x0_hat.device
        n_total = qb.num_nodes
        if n_total == 0:
            return torch.tensor(0.0, device=device)

        # 그래프 단위로 평균
        total = x0_hat.new_zeros(())
        count = x0_hat.new_zeros(())
        upto = int(self.repulsion_exclude_hops)

        # 간단한 VDW 상한 (원본과 동일 딕셔너리)
        VDW = {1:1.20, 6:1.70, 7:1.55, 8:1.52, 9:1.47, 15:1.80, 16:1.80, 17:1.75, 35:1.85, 53:1.98}
        def _vdw_sum(zi, zj):
            ri = VDW.get(int(zi), 1.70); rj = VDW.get(int(zj), 1.70)
            return ri + rj

        # 원자번호
        if hasattr(qb, 'Z') and qb.Z is not None:
            Z_all = qb.Z.long()
        else:
            Z_all = torch.clamp(qb.atom_type.long(), 1, 118)

        for g in range(qb.num_graphs):
            idx = (qb.batch == g).nonzero(as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n < 2:
                continue

            # 로컬 edge_index
            g2l = torch.full((qb.pos.size(0),), -1, dtype=torch.long, device=device)
            g2l[idx] = torch.arange(n, device=device)
            ei = qb.edge_index
            mask = (g2l[ei[0]] >= 0) & (g2l[ei[1]] >= 0)
            li = g2l[ei[0][mask]]; lj = g2l[ei[1][mask]]
            local_ei = torch.stack([li, lj], dim=0) if li.numel() > 0 else None

            # upto-hop 제외 마스크
            excl = self._exclude_mask_hops(local_ei, n, upto=upto, device=device)
            D = torch.cdist(x0_hat[idx], x0_hat[idx], p=2)                      # [n,n]
            iu = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
            pair_mask = iu & (~excl)
            if not torch.any(pair_mask):
                continue
            I, J = pair_mask.nonzero(as_tuple=True)
            d = D[I, J]

            Zi, Zj = Z_all[idx][I], Z_all[idx][J]
            with torch.no_grad():
                m_np = [_vdw_sum(int(zi.item()), int(zj.item())) for zi, zj in zip(Zi, Zj)]
            margin = torch.tensor(m_np, device=device, dtype=torch.float32) * 0.85 - 0.10
            margin = torch.clamp(margin, min=max(0.50, float(self.repulsion_margin)))  # 안전 하한

            rep = (F.relu(margin - d) ** 2).mean()
            total += rep; count += 1.0

        return (total / torch.clamp(count, min=1.0))

    @staticmethod
    def _exclude_mask_hops(local_edge_index: Optional[torch.Tensor], n: int, upto: int = 3, device='cpu'):
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        if local_edge_index is not None and local_edge_index.numel() > 0:
            i, j = local_edge_index[0], local_edge_index[1]
            adj[i, j] = True; adj[j, i] = True
        excl = adj.clone()
        if upto >= 2:
            A2 = (adj.float() @ adj.float()) > 0; A2.fill_diagonal_(False); excl |= A2
        if upto >= 3:
            A2 = (adj.float() @ adj.float()) > 0; A2.fill_diagonal_(False)
            A3 = (A2.float() @ adj.float()) > 0; A3.fill_diagonal_(False); excl |= A3
        iu = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
        return iu & excl

    # -------------- Sampler --------------

    @torch.no_grad()
    def FM_Sampling(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        num_steps: int = None,
        cfg_scale: float = 1.0,
        # ---- UFF options ----
        query_mols=None,
        pocket_mols=None,
        uff_guidance_scale: float = 0.0,
        uff_inner_steps: int = 8,
        uff_clamp: float = 1.0,
        uff_start_ratio: float = 0.0,
        snr_gate_gamma: float = 1.0,
        uff_vdw_multiplier: float = 10.0,
        # ---- Temperature ----
        noise_temperature: float = 0.0,   # 0 = deterministic ODE
        debug_log: bool = False,
    ):
        """
        Euler ODE integration for flow matching.
        - t: 0 (noise) -> 1 (data)
        - dx/dt = v_hat(x_t, t)
        - x0_pred = x_t + (1-t) * v_hat
        - UFF/Vina steering plugs in on x0_pred exactly as in DiffAlign
        """
        import math

        device = next(self.parameters()).device
        qb = query_batch.to(device)
        rb = reference_batch.to(device)

        T = num_steps if num_steps is not None else self.num_steps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # query node indices
        mol_slices_q = [(qb.batch == i).nonzero(as_tuple=True)[0] for i in range(qb.num_graphs)]
        gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)
        B = qb.num_graphs
        Nq = mol_slices_q[0].numel()

        # UFF setup (identical to DiffAlign)
        use_uff = (uff_guidance_scale > 0.0 and query_mols is not None and pocket_mols is not None)
        if use_uff:
            from uff_torch import UFFTorch, build_uff_inputs, merge_uff_inputs
            q_inputs = build_uff_inputs(query_mols, device=device, dtype=torch.float32,
                                        vdw_distance_multiplier=uff_vdw_multiplier,
                                        ignore_interfragment_interactions=False)
            p_inputs = build_uff_inputs(pocket_mols, device=device, dtype=torch.float32,
                                        vdw_distance_multiplier=uff_vdw_multiplier,
                                        ignore_interfragment_interactions=False)
            qp_inputs = merge_uff_inputs(q_inputs, p_inputs,
                                         ignore_interfragment_interactions=False,
                                         vdw_distance_multiplier=float(uff_vdw_multiplier))
            uff_model = UFFTorch(qp_inputs).to(device).eval()

            def _mol_coords(m):
                conf = m.GetConformer()
                return torch.tensor([[conf.GetAtomPosition(k).x,
                                      conf.GetAtomPosition(k).y,
                                      conf.GetAtomPosition(k).z]
                                     for k in range(m.GetNumAtoms())],
                                    device=device, dtype=torch.float32)
            pocket_coords_fixed = torch.stack([_mol_coords(m) for m in pocket_mols], dim=0)
        else:
            uff_model = None
            pocket_coords_fixed = None

        # initialize from noise at t=0
        x_t = torch.randn((qb.num_nodes, 3), device=device, dtype=torch.float32)

        dt = 1.0 / T  # step size

        for i in range(T):
            t = i / T                          # current t in [0, 1)
            t_graph = torch.full((qb.num_graphs,), t, device=device, dtype=torch.float32)

            cur_q = qb.clone()
            cur_q.pos = x_t

            # predict velocity
            if cfg_scale == 1.0:
                v_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_merged = v_u + cfg_scale * (v_c - v_u)

            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_merged[qmask]

            # x0 prediction: x0 = x_t + (1-t) * v_hat
            x0_pred = x_t + (1.0 - t) * v_hat

            # SNR-aware gate (reuse same formula as DiffAlign)
            sigma_t = 1.0 - t   # noise level decreases as t increases
            gate_t = (max(0.0, min(1.0, 1.0 - sigma_t))) ** float(snr_gate_gamma)

            # UFF steering on x0_pred
            x0_star = x0_pred
            if use_uff and (t / max(1e-8, 1.0)) >= uff_start_ratio and gate_t > 0.0:
                from torch import autograd
                coords_q0 = x0_star.index_select(0, gather_idx_q).view(B, Nq, 3).detach()
                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_q0)
                coords_q = coords_q0.clone()
                inner = max(1, int(uff_inner_steps))
                step_scale = (uff_guidance_scale * gate_t) / float(inner)
                for _ in range(inner):
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)
                        coords_cat = torch.cat([coords_q_req, pocket_coords_fixed], dim=1)
                        E_b = uff_model(coords_cat)
                        if E_b.ndim == 0: E_b = E_b.unsqueeze(0)
                        grad_q, = autograd.grad(E_b.sum(), coords_q_req)
                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                    coords_q = (coords_q + step_scale * forces_q).detach()
                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q.reshape(B * Nq, 3))

            # Euler step toward x0_star
            # dx = v_hat * dt, but we use x0_star instead of x0_pred
            v_star = (x0_star - x_t) / max(1.0 - t, 1e-8)
            x_t = x_t + dt * v_star

            # optional stochastic noise (SDE mode, default off)
            if noise_temperature > 0.0 and i < T - 1:
                x_t = x_t + math.sqrt(dt) * noise_temperature * torch.randn_like(x_t)

        return (x_t, None)
