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

class DiffAlign(nn.Module):
    """
    Isotropic Gaussian Diffusion (v-parameterization; T steps)
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

        # Diffusion
        num_timesteps: int = 32,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'cosine',

        # Repulsion
        repulsion_weight: float = 1e-2,
        repulsion_margin: float = 1.2,
        repulsion_exclude_hops: int = 3,
    ):
        super().__init__()

        # ---- Diffusion buffers (isotropic) ----
        self.num_timesteps = int(num_timesteps)
        if schedule_type == 'linear':
            betas = linear_beta_schedule(self.num_timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))

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
        v-타깃 학습 + x0 절대좌표 앵커 + repulsion(옵션)
        - Isotropic: 그래프별 스텝 t에 대해 ᾱ_t, (1-ᾱ_t)를 스칼라로 사용
        - v_target = ᾱ^0.5 * ε − (1-ᾱ)^0.5 * x0
        """
        if query_batch.num_nodes == 0:
            device = reference_batch.pos.device if hasattr(reference_batch, 'pos') and reference_batch.num_nodes > 0 else 'cpu'
            return torch.tensor(0.0, device=device, requires_grad=True)

        device = query_batch.pos.device
        x0 = query_batch.pos
        num_graphs = query_batch.num_graphs
        T = self.num_timesteps

        # 그래프별 랜덤 타임스텝
        t_graph = torch.randint(0, T, (num_graphs,), device=device).long()

        # 그래프별 스칼라 ᾱ, σ
        abar_g = self.sqrt_alphas_cumprod[t_graph]                              # [G]
        sig_g  = self.sqrt_one_minus_alphas_cumprod[t_graph]                    # [G]

        # 노드로 브로드캐스트
        abar_n = abar_g[query_batch.batch].unsqueeze(1)                         # [Nq,1]
        sig_n  = sig_g[query_batch.batch].unsqueeze(1)                          # [Nq,1]

        eps = torch.randn_like(x0)
        x_t = abar_n * x0 + sig_n * eps                                         # [Nq,3]
        v_target = abar_n * eps - sig_n * x0                                    # [Nq,3]

        noisy_query = query_batch.clone(); noisy_query.pos = x_t

        # v 예측
        if self.training and (torch.rand(1, device=device) < 0.2):
            v_pred_merged = self(noisy_query, reference_batch, t_graph, condition=False)
        else:
            v_pred_merged = self(noisy_query, reference_batch, t_graph, condition=True)

        merged = merge_graphs_in_batch(noisy_query, reference_batch)
        qmask = (merged.graph_idx % 2 == 0)
        v_pred = v_pred_merged[qmask]                                           # [Nq,3]

        # v-loss
        v_loss = F.mse_loss(v_pred, v_target, reduction='mean')

        # x0 anchor (절대 포즈 회귀)
        x0_hat = abar_n * x_t - sig_n * v_pred
        x0_loss = F.mse_loss(x0_hat, x0, reduction='mean')

        # repulsion (비결합 쌍 충돌 억제; 간단 평균)
        rep_loss = self._repulsion_loss(query_batch, x0_hat)

        w_v   = float(getattr(self, 'v_loss_weight',   1.0))
        w_x0  = float(getattr(self, 'x0_loss_weight',  1.0))
        w_rep = float(getattr(self, 'repulsion_weight', self.repulsion_weight))
        loss = w_v*v_loss + w_x0*x0_loss + w_rep*rep_loss
        # loss = w_v*v_loss + w_x0*x0_loss
        return loss

    @torch.no_grad()
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

    # -------------- Samplers --------------

    @torch.no_grad()
    def DDPM_Sampling_Heun_UFF(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        clamp: float = 1e-10,
        cfg_scale: float = 1.0,
        # ---- UFF(PyTorch) 옵션 ----
        query_mols=None,                      # RDKit Mol 리스트 (len == num_graphs). None이면 UFF off
        pocket_mols=None,                     # RDKit Mol 리스트 (len == num_graphs). None이면 포켓 off
        uff_guidance_scale: float = 0.0,      # 0 → UFF 비활성
        uff_inner_steps: int = 8,             # UFF 내부 gradient step 횟수
        uff_clamp: float = 1.0,               # 힘 클램프
        uff_start_ratio: float = 0.0,         # (t/(T-1)) < ratio 이면 UFF skip
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        snr_use_zero_mode_mask: bool = False, # (isotropic 버전에서는 사용 안 함; 호환용)
        x0_fuse_corrector_weight: float = 0.5,# x0_fused = (1-w)*x0_pred + w*x0_corr
        # ---- Temperature 옵션 ----
        noise_temperature: float = 0.3,       # posterior 노이즈 온도 τ
        # ---- UFF 동적 nonbonded 파라미터 ----
        uff_vdw_multiplier: float = 10.0,     # 동적 컷오프 배수 (vdw_distance_multiplier)
        debug_log: bool = False,
    ):
        """
        Heun predictor-corrector (v-param, isotropic) + UFFTorch 기반 UFF steering.

        - 디퓨전 부분: x-공간에서 v-param DDPM Heun (predictor-corrector).
        - UFF 부분: torch-native UFFTorch를 사용해 batched 리간드-포켓 에너지/gradient 계산.
          * query_mols, pocket_mols → build_uff_inputs → merge_uff_inputs → UFFTorch(qp_inputs)
          * 각 스텝에서 x0_fused를 기준으로 리간드 좌표만 UFF gradient로 여러 번 업데이트.
          * 최종 coords_q를 x0_fused에 커밋하여 x0_star로 사용.
        - SNR-aware gate: σ_t = sqrt(1 - abar_t), h(t) = 1 - σ_t, gate(t) = h(t)^γ
          (논문에서 설명한 late–strong 스케줄과 일치).
        """
        import math
        import torch
        from torch import autograd

        # 외부에서 제공되는 UFF 유틸 함수/클래스가 있다고 가정
        # from uff_torch import build_uff_inputs, merge_uff_inputs, UFFTorch

        device = self.betas.device

        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- UFFTorch 세팅 ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs, (
                f"len(query_mols)={len(query_mols)} != qb.num_graphs={qb.num_graphs}"
            )
            assert len(pocket_mols) == qb.num_graphs, (
                f"len(pocket_mols)={len(pocket_mols)} != qb.num_graphs={qb.num_graphs}"
            )

            # RDKit Mol → UFFTorch 입력
            q_inputs_ref = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )
            p_inputs_ref = build_uff_inputs(
                pocket_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            qp_inputs = merge_uff_inputs(
                q_inputs_ref,
                p_inputs_ref,
                ignore_interfragment_interactions=False,
                vdw_distance_multiplier=float(uff_vdw_multiplier),
            )

            uff_model = UFFTorch(qp_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            # query 노드 인덱스 (Batch → per-mol slice)
            mol_slices_q = [
                (qb.batch == i).nonzero(as_tuple=True)[0]
                for i in range(qb.num_graphs)
            ]
            gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)  # [ΣNq]

            # 단순화를 위해 Nq, Np를 모두 동일하다고 가정 (CrossDocked에서 보통 맞춰둠)
            B = qb.num_graphs
            Nq = mol_slices_q[0].numel()
            for sl in mol_slices_q:
                assert sl.numel() == Nq, "현재 구현은 모든 query에 동일한 Nq를 가정합니다."

            # pocket coords (고정)
            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [
                        [
                            conf.GetAtomPosition(k).x,
                            conf.GetAtomPosition(k).y,
                            conf.GetAtomPosition(k).z,
                        ]
                        for k in range(m.GetNumAtoms())
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols],
                dim=0,
            )  # [B,Np,3]
            Np = pocket_coords_fixed.shape[1]

        else:
            uff_model = None

        # ---------- x_T 초기화 (표준 N(0,I)) ----------
        x_t = torch.randn(
            (qb.num_nodes, 3),
            device=device,
            dtype=torch.float32,
        )

        # ---------- 메인 역과정 루프 ----------
        for t in reversed(range(T)):
            t_graph = torch.full(
                (qb.num_graphs,),
                t,
                device=device,
                dtype=torch.long,
            )

            # 현재 좌표를 query batch에 세팅
            cur_q = qb.clone()
            cur_q.pos = x_t

            # v_hat@t (CFG)
            if cfg_scale == 1.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_scale * (v_c - v_u)

            # merge해서 query 노드만 선택
            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]  # [Nq_total,3] == qb.num_nodes

            # 계수 (스칼라)
            abar_t = self.sqrt_alphas_cumprod[t]          # sqrt(abar_t)
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t]# sigma_t = sqrt(1-abar_t)

            # 1) x0_pred 복원 (v-param: x0 = √ᾱ x_t − √(1-ᾱ) v_t)
            x0_pred = abar_t * x_t - sig_t * v_hat

            # 2) posterior mean μ_t(x0_pred) → x_{t-1}^{euler}
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_tm1_euler = c1_t * x0_pred + c2_t * x_t

            # 3) corrector: t>0이면 v_hat@t-1 on x_{t-1}^{euler} → x0_corr
            if t > 0:
                t_graph2 = torch.full(
                    (qb.num_graphs,),
                    t - 1,
                    device=device,
                    dtype=torch.long,
                )
                cur_q2 = qb.clone()
                cur_q2.pos = x_tm1_euler

                if cfg_scale == 1.0:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=True)
                else:
                    v_u2 = self(cur_q2, rb, t_graph2, condition=False)
                    v_c2 = self(cur_q2, rb, t_graph2, condition=True)
                    v_hat2_merged = v_u2 + cfg_scale * (v_c2 - v_u2)

                merged2 = merge_graphs_in_batch(cur_q2, rb, device=device)
                qmask2 = (merged2.graph_idx % 2 == 0)
                v_hat2 = v_hat2_merged[qmask2]

                abar_t1 = self.sqrt_alphas_cumprod[t - 1]
                sig_t1  = self.sqrt_one_minus_alphas_cumprod[t - 1]

                x0_corr = abar_t1 * x_tm1_euler - sig_t1 * v_hat2
            else:
                x0_corr = None

            # 4) x0_fused = (1-w)*x0_pred + w*x0_corr
            if x0_corr is not None:
                w = float(x0_fuse_corrector_weight)
                x0_fused = (1.0 - w) * x0_pred + w * x0_corr
            else:
                x0_fused = x0_pred

            # 5) SNR-aware gate (h(t) = 1 - σ_t, gate = h^γ)
            sigma_t = float(sig_t.item())
            h_t = 1.0 - sigma_t
            h_t = max(0.0, min(1.0, h_t))
            gate_t = h_t ** float(snr_gate_gamma)

            # 6) UFF로 x0_fused를 보정 → x0_star
            if (
                use_uff
                and (t / max(1, T - 1)) >= uff_start_ratio
                and gate_t > 0.0
            ):
                # 현재 x0_fused에서 query 좌표만 [B,Nq,3]으로 reshape
                coords_q0 = x0_fused.index_select(0, gather_idx_q)  # [ΣNq,3]
                coords_q0 = coords_q0.view(B, Nq, 3).detach()

                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_q0)

                coords_q = coords_q0.clone()

                inner = max(1, int(uff_inner_steps))
                # 여러 번 나눠서 업데이트
                step_scale = (uff_guidance_scale * gate_t) / float(inner)

                for _ in range(inner):
                    coords_q_req = coords_q.clone().requires_grad_(True)  # [B,Nq,3]
                    coords_cat = torch.cat(
                        [coords_q_req, pocket_coords_fixed],
                        dim=1,
                    )  # [B,Nq+Np,3]

                    E_b = uff_model(coords_cat)  # (B,) or scalar
                    if not isinstance(E_b, torch.Tensor):
                        E_b = torch.as_tensor(E_b, device=device, dtype=coords_cat.dtype)
                    if E_b.ndim == 0:
                        E_b = E_b.unsqueeze(0)

                    grad_q, = autograd.grad(
                        E_b.sum(),
                        coords_q_req,
                        create_graph=False,
                        retain_graph=False,
                    )

                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)  # [B,Nq,3]
                    coords_q = (coords_q + step_scale * forces_q).detach()

                # 최종 coords_q를 x0_fused에 커밋 → x0_star
                x0_star = x0_fused.clone()
                x0_star.index_copy_(
                    0,
                    gather_idx_q,
                    coords_q.reshape(B * Nq, 3),
                )

                if debug_log:
                    f_norm = forces_q.norm(dim=-1)
                    print(
                        f"[UFF] t={t:02d} gate={gate_t:.3f} | "
                        f"‖F‖ mean={f_norm.mean().item():.3f}, "
                        f"max={f_norm.max().item():.3f}"
                    )
            else:
                x0_star = x0_fused

            # 7) μ_t(x0_star), μ_{t-1}(x0_star) → Heun 평균
            x_mu_t = c1_t * x0_star + c2_t * x_t

            if t > 0:
                c1_t1 = self.posterior_mean_coef1[t - 1]
                c2_t1 = self.posterior_mean_coef2[t - 1]
                x_mu_tm1 = c1_t1 * x0_star + c2_t1 * x_tm1_euler
                x_next_det = 0.5 * (x_mu_t + x_mu_tm1)
            else:
                x_next_det = x_mu_t

            # 8) posterior noise (+ temperature)
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)

                noise = torch.randn_like(x_t)
                x_t = x_next_det + math.sqrt(var_t) * noise
            else:
                x_t = x_next_det

        return (x_t, None)

    @torch.no_grad()
    def DDPM_Sampling_Heun_UFF_vina(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        cfg_scale: float = 1.0,
        # ---- UFF(PyTorch) 옵션 ----
        query_mols=None,                      # RDKit Mol 리스트
        pocket_mols=None,                     # RDKit Mol 리스트
        uff_guidance_scale: float = 0.0,      # 0 → UFF 비활성
        uff_inner_steps: int = 8,
        uff_clamp: float = 1.0,
        uff_start_ratio: float = 0.0,         # (t/(T-1)) < ratio 이면 UFF/Vina skip
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        x0_fuse_corrector_weight: float = 0.5,# x0_fused = (1-w)*x0_pred + w*x0_corr
        # ---- Temperature 옵션 ----
        noise_temperature: float = 0.3,       # posterior 노이즈 온도
        # ---- UFF 동적 nonbonded 파라미터 ----
        uff_vdw_multiplier: float = 10.0,
        debug_log: bool = False,
        # ---- VinaSF 옵션 ----
        vina_guidance_scale: float = 0.0,     # 0 → Vina 비활성
        vina_inner_steps: int = 1,
        vina_clamp: float = 1.0,
        vina_perm_qb2rd=None,                 # (Nq,) permutation
    ):
        """
        Fixed Heun predictor-corrector + UFF/Vina Steering.
        
        Flow:
        1. Predictor: x_t -> x0_pred (via v_hat_t)
        2. Euler Step: x0_pred -> x_{t-1}_euler
        3. Corrector: x_{t-1}_euler -> x0_corr (via v_hat_{t-1})
        4. Fusion: x0_fused = Mix(x0_pred, x0_corr)
        5. Physics Guidance: x0_fused -> Optimize via UFF/Vina -> x0_star
        6. Final Transition: Compute μ(x_t, x0_star) -> x_{t-1}
        """

        device = self.betas.device
        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- 공통: query 노드 슬라이스 / 인덱스 ----------
        mol_slices_q = [
            (qb.batch == i).nonzero(as_tuple=True)[0]
            for i in range(qb.num_graphs)
        ]
        gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)  # [ΣNq]

        B = qb.num_graphs
        Nq = mol_slices_q[0].numel()
        for sl in mol_slices_q:
            assert sl.numel() == Nq, "현재 구현은 모든 query에 동일한 Nq를 가정합니다."

        # ---------- UFFTorch 세팅 ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs
            assert len(pocket_mols) == qb.num_graphs

            q_inputs = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            p_inputs = build_uff_inputs(
                pocket_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            qp_inputs = merge_uff_inputs(
                q_inputs,
                p_inputs,
                ignore_interfragment_interactions=False,
                vdw_distance_multiplier=float(uff_vdw_multiplier),
            )

            uff_model = UFFTorch(qp_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [
                        [
                            conf.GetAtomPosition(k).x,
                            conf.GetAtomPosition(k).y,
                            conf.GetAtomPosition(k).z,
                        ]
                        for k in range(m.GetNumAtoms())
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols],
                dim=0,
            )  # [B,Np,3]
        else:
            uff_model = None
            pocket_coords_fixed = None

        # ---------- VinaSF 세팅 ----------
        use_vina = (vina_guidance_scale > 0.0)
        if use_vina:
            vina_model = VinaSFTorch.from_rdkit(pocket_mols[0], query_mols[0]).to(device).eval()
            if vina_perm_qb2rd is not None:
                vina_perm_qb2rd = vina_perm_qb2rd.to(device)

        # ---------- x_T 초기화 ----------
        x_t = torch.randn((qb.num_nodes, 3), device=device, dtype=torch.float32)

        # ---------- 메인 역과정 루프 ----------
        for t in reversed(range(T)):
            t_graph = torch.full((qb.num_graphs,), t, device=device, dtype=torch.long)

            # 1) Predictor: x_t -> v_hat -> x0_pred
            cur_q = qb.clone()
            cur_q.pos = x_t

            if cfg_scale == 1.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_scale * (v_c - v_u)

            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]

            abar_t = self.sqrt_alphas_cumprod[t]
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t]

            x0_pred = abar_t * x_t - sig_t * v_hat

            # 2) Tentative Euler Step for Corrector: x0_pred -> x_{t-1}^{euler}
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_tm1_euler = c1_t * x0_pred + c2_t * x_t

            # 3) Corrector: Check x_{t-1} -> x0_corr
            x0_corr = None
            if t > 0:
                t_graph2 = torch.full((qb.num_graphs,), t - 1, device=device, dtype=torch.long)
                cur_q2 = qb.clone()
                cur_q2.pos = x_tm1_euler

                if cfg_scale == 1.0:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=True)
                else:
                    v_u2 = self(cur_q2, rb, t_graph2, condition=False)
                    v_c2 = self(cur_q2, rb, t_graph2, condition=True)
                    v_hat2_merged = v_u2 + cfg_scale * (v_c2 - v_u2)

                merged2 = merge_graphs_in_batch(cur_q2, rb, device=device)
                qmask2 = (merged2.graph_idx % 2 == 0)
                v_hat2 = v_hat2_merged[qmask2]

                # Note: Using coefficients at t-1 to revert from x_{t-1}
                abar_t1 = self.sqrt_alphas_cumprod[t - 1]
                sig_t1  = self.sqrt_one_minus_alphas_cumprod[t - 1]
                
                x0_corr = abar_t1 * x_tm1_euler - sig_t1 * v_hat2

            # 4) Fusion: x0_fused
            if x0_corr is not None:
                w = float(x0_fuse_corrector_weight)
                x0_fused = (1.0 - w) * x0_pred + w * x0_corr
            else:
                x0_fused = x0_pred

            # 5) Physics Guidance (Steering on x0_fused)
            sigma_t = float(sig_t.item())
            gate_t = (max(0.0, min(1.0, 1.0 - sigma_t))) ** float(snr_gate_gamma)

            x0_star = x0_fused

            # 5-1) Vina Steering
            if use_vina and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0_v = x0_star.index_select(0, gather_idx_q)
                coords_q0_v = coords_q0_v.view(B, Nq, 3).contiguous().detach()
                coords_q_v = coords_q0_v.clone()

                inner_v = max(1, int(vina_inner_steps))
                step_v = (float(vina_guidance_scale) * gate_t) / float(inner_v)

                for _ in range(inner_v):
                    coords_req = coords_q_v.clone()
                    coords_in = coords_req[:, vina_perm_qb2rd, :] if vina_perm_qb2rd is not None else coords_req
                    
                    with torch.enable_grad():
                        score, grad = vina_model.score_and_gradient(coords_in)

                    if not isinstance(grad, torch.Tensor):
                        grad = torch.as_tensor(grad, device=device, dtype=coords_in.dtype)
                    if grad.ndim == 2: grad = grad.unsqueeze(0)

                    if vina_perm_qb2rd is not None:
                        grad_q_v = torch.zeros_like(coords_req)
                        grad_q_v[:, vina_perm_qb2rd, :] = grad
                    else:
                        grad_q_v = grad

                    forces_v = (-grad_q_v).detach().clamp_(-vina_clamp, vina_clamp)
                    coords_q_v = (coords_q_v + step_v * forces_v).detach()

                # Apply Vina result
                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q_v.view(B * Nq, 3))

                if debug_log:
                    print(f"[Vina] t={t} ForceNorm={forces_v.norm(dim=-1).mean():.3f}")

            # 5-2) UFF Steering
            if use_uff and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0 = x0_star.index_select(0, gather_idx_q)
                coords_q0 = coords_q0.view(B, Nq, 3).contiguous().detach()
                coords_0 = torch.cat([coords_q0, pocket_coords_fixed], dim=1)

                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_0)

                coords_q = coords_q0.clone()
                inner = max(1, int(uff_inner_steps))
                step_scale = (uff_guidance_scale * gate_t) / float(inner)

                for _ in range(inner):
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)
                        coords_cat = torch.cat([coords_q_req, pocket_coords_fixed], dim=1)
                        E_b = uff_model(coords_cat)
                        if E_b.ndim == 0: E_b = E_b.unsqueeze(0)
                        grad_q, = torch.autograd.grad(E_b.sum(), coords_q_req)

                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                    coords_q = (coords_q + step_scale * forces_q).detach()

                # Apply UFF result
                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q.view(B * Nq, 3))
                
                if debug_log:
                    print(f"[UFF] t={t} ForceNorm={forces_q.norm(dim=-1).mean():.3f}")

            # 6) [수정됨] Final Transition: Compute μ_t using x0_star
            # 이전 코드의 잘못된 평균화 로직(t -> t-1, t-1 -> t-2 혼용)을 삭제함.
            # 대신 물리적으로 보정된 x0_star를 이용해 DDPM Posterior Mean을 계산.
            # μ_t = c1_t * x0_recon + c2_t * x_t
            
            x_mean = c1_t * x0_star + c2_t * x_t
            
            # 7) Noise Addition (Langevin Dynamics / Standard DDPM sampling)
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)
                
                noise = torch.randn_like(x_t)
                x_t = x_mean + math.sqrt(var_t) * noise
            else:
                x_t = x_mean

        return (x_t, None)


    @torch.no_grad()
    def DDPM_Sampling_Heun_UFF_vina_wo_scheduling(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        cfg_scale: float = 1.0,
        # ---- UFF(PyTorch) 옵션 ----
        query_mols=None,                      # RDKit Mol 리스트
        pocket_mols=None,                     # RDKit Mol 리스트
        uff_guidance_scale: float = 0.0,      # 0 → UFF 비활성
        uff_inner_steps: int = 8,
        uff_clamp: float = 1.0,
        uff_start_ratio: float = 0.0,         # (t/(T-1)) < ratio 이면 UFF/Vina skip
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        x0_fuse_corrector_weight: float = 0.5,# x0_fused = (1-w)*x0_pred + w*x0_corr
        # ---- Temperature 옵션 ----
        noise_temperature: float = 0.3,       # posterior 노이즈 온도
        # ---- UFF 동적 nonbonded 파라미터 ----
        uff_vdw_multiplier: float = 10.0,
        debug_log: bool = False,
        # ---- VinaSF 옵션 ----
        vina_guidance_scale: float = 0.0,     # 0 → Vina 비활성
        vina_inner_steps: int = 1,
        vina_clamp: float = 1.0,
        vina_perm_qb2rd=None,                 # (Nq,) permutation
    ):
        """
        Fixed Heun predictor-corrector + UFF/Vina Steering.
        
        Flow:
        1. Predictor: x_t -> x0_pred (via v_hat_t)
        2. Euler Step: x0_pred -> x_{t-1}_euler
        3. Corrector: x_{t-1}_euler -> x0_corr (via v_hat_{t-1})
        4. Fusion: x0_fused = Mix(x0_pred, x0_corr)
        5. Physics Guidance: x0_fused -> Optimize via UFF/Vina -> x0_star
        6. Final Transition: Compute μ(x_t, x0_star) -> x_{t-1}
        """

        device = self.betas.device
        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- 공통: query 노드 슬라이스 / 인덱스 ----------
        mol_slices_q = [
            (qb.batch == i).nonzero(as_tuple=True)[0]
            for i in range(qb.num_graphs)
        ]
        gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)  # [ΣNq]

        B = qb.num_graphs
        Nq = mol_slices_q[0].numel()
        for sl in mol_slices_q:
            assert sl.numel() == Nq, "현재 구현은 모든 query에 동일한 Nq를 가정합니다."

        # ---------- UFFTorch 세팅 ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs
            assert len(pocket_mols) == qb.num_graphs

            q_inputs = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            p_inputs = build_uff_inputs(
                pocket_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            qp_inputs = merge_uff_inputs(
                q_inputs,
                p_inputs,
                ignore_interfragment_interactions=False,
                vdw_distance_multiplier=float(uff_vdw_multiplier),
            )

            uff_model = UFFTorch(qp_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [
                        [
                            conf.GetAtomPosition(k).x,
                            conf.GetAtomPosition(k).y,
                            conf.GetAtomPosition(k).z,
                        ]
                        for k in range(m.GetNumAtoms())
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols],
                dim=0,
            )  # [B,Np,3]
        else:
            uff_model = None
            pocket_coords_fixed = None

        # ---------- VinaSF 세팅 ----------
        use_vina = (vina_guidance_scale > 0.0)
        if use_vina:
            vina_model = VinaSFTorch.from_rdkit(pocket_mols[0], query_mols[0]).to(device).eval()
            if vina_perm_qb2rd is not None:
                vina_perm_qb2rd = vina_perm_qb2rd.to(device)

        # ---------- x_T 초기화 ----------
        x_t = torch.randn((qb.num_nodes, 3), device=device, dtype=torch.float32)

        # ---------- 메인 역과정 루프 ----------
        for t in reversed(range(T)):
            t_graph = torch.full((qb.num_graphs,), t, device=device, dtype=torch.long)

            # 1) Predictor: x_t -> v_hat -> x0_pred
            cur_q = qb.clone()
            cur_q.pos = x_t

            if cfg_scale == 1.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_scale * (v_c - v_u)

            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]

            abar_t = self.sqrt_alphas_cumprod[t]
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t]

            x0_pred = abar_t * x_t - sig_t * v_hat

            # 2) Tentative Euler Step for Corrector: x0_pred -> x_{t-1}^{euler}
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_tm1_euler = c1_t * x0_pred + c2_t * x_t

            # 3) Corrector: Check x_{t-1} -> x0_corr
            x0_corr = None
            if t > 0:
                t_graph2 = torch.full((qb.num_graphs,), t - 1, device=device, dtype=torch.long)
                cur_q2 = qb.clone()
                cur_q2.pos = x_tm1_euler

                if cfg_scale == 1.0:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=True)
                else:
                    v_u2 = self(cur_q2, rb, t_graph2, condition=False)
                    v_c2 = self(cur_q2, rb, t_graph2, condition=True)
                    v_hat2_merged = v_u2 + cfg_scale * (v_c2 - v_u2)

                merged2 = merge_graphs_in_batch(cur_q2, rb, device=device)
                qmask2 = (merged2.graph_idx % 2 == 0)
                v_hat2 = v_hat2_merged[qmask2]

                # Note: Using coefficients at t-1 to revert from x_{t-1}
                abar_t1 = self.sqrt_alphas_cumprod[t - 1]
                sig_t1  = self.sqrt_one_minus_alphas_cumprod[t - 1]
                
                x0_corr = abar_t1 * x_tm1_euler - sig_t1 * v_hat2

            # 4) Fusion: x0_fused
            if x0_corr is not None:
                w = float(x0_fuse_corrector_weight)
                x0_fused = (1.0 - w) * x0_pred + w * x0_corr
            else:
                x0_fused = x0_pred

            # 5) Physics Guidance (Steering on x0_fused)
            sigma_t = float(sig_t.item())
            gate_t = (max(0.0, min(1.0, 1.0 - sigma_t))) ** float(snr_gate_gamma)

            x0_star = x0_fused

            # 5-1) Vina Steering
            if use_vina and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0_v = x0_star.index_select(0, gather_idx_q)
                coords_q0_v = coords_q0_v.view(B, Nq, 3).contiguous().detach()
                coords_q_v = coords_q0_v.clone()

                inner_v = max(1, int(vina_inner_steps))
                # step_v = (float(vina_guidance_scale) * gate_t) / float(inner_v)

                for _ in range(inner_v):
                    coords_req = coords_q_v.clone()
                    coords_in = coords_req[:, vina_perm_qb2rd, :] if vina_perm_qb2rd is not None else coords_req
                    
                    with torch.enable_grad():
                        score, grad = vina_model.score_and_gradient(coords_in)

                    if not isinstance(grad, torch.Tensor):
                        grad = torch.as_tensor(grad, device=device, dtype=coords_in.dtype)
                    if grad.ndim == 2: grad = grad.unsqueeze(0)

                    if vina_perm_qb2rd is not None:
                        grad_q_v = torch.zeros_like(coords_req)
                        grad_q_v[:, vina_perm_qb2rd, :] = grad
                    else:
                        grad_q_v = grad

                    forces_v = (-grad_q_v).detach().clamp_(-vina_clamp, vina_clamp)
                    coords_q_v = (coords_q_v + vina_guidance_scale * forces_v).detach()

                # Apply Vina result
                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q_v.view(B * Nq, 3))

                if debug_log:
                    print(f"[Vina] t={t} ForceNorm={forces_v.norm(dim=-1).mean():.3f}")

            # 5-2) UFF Steering
            if use_uff and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0 = x0_star.index_select(0, gather_idx_q)
                coords_q0 = coords_q0.view(B, Nq, 3).contiguous().detach()
                coords_0 = torch.cat([coords_q0, pocket_coords_fixed], dim=1)

                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_0)

                coords_q = coords_q0.clone()
                inner = max(1, int(uff_inner_steps))
                # step_scale = (uff_guidance_scale * gate_t) / float(inner)

                for _ in range(inner):
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)
                        coords_cat = torch.cat([coords_q_req, pocket_coords_fixed], dim=1)
                        E_b = uff_model(coords_cat)
                        if E_b.ndim == 0: E_b = E_b.unsqueeze(0)
                        grad_q, = torch.autograd.grad(E_b.sum(), coords_q_req)

                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                    coords_q = (coords_q + uff_guidance_scale * forces_q).detach()

                # Apply UFF result
                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q.view(B * Nq, 3))
                
                if debug_log:
                    print(f"[UFF] t={t} ForceNorm={forces_q.norm(dim=-1).mean():.3f}")

            # 6) [수정됨] Final Transition: Compute μ_t using x0_star
            # 이전 코드의 잘못된 평균화 로직(t -> t-1, t-1 -> t-2 혼용)을 삭제함.
            # 대신 물리적으로 보정된 x0_star를 이용해 DDPM Posterior Mean을 계산.
            # μ_t = c1_t * x0_recon + c2_t * x_t
            
            x_mean = c1_t * x0_star + c2_t * x_t
            
            # 7) Noise Addition (Langevin Dynamics / Standard DDPM sampling)
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)
                
                noise = torch.randn_like(x_t)
                x_t = x_mean + math.sqrt(var_t) * noise
            else:
                x_t = x_mean

        return (x_t, None)

    @torch.no_grad()
    def DDPM_Sampling_Heun_UFF_vina_cfg_scheduling(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        cfg_scale: float = 1.0,
        cfg_floor: float = 0.0,               # NEW: 후반부 CFG 하한 (cosine으로 cfg_scale -> cfg_floor)
        # ---- UFF(PyTorch) 옵션 ----
        query_mols=None,                      # RDKit Mol 리스트
        pocket_mols=None,                     # RDKit Mol 리스트
        uff_guidance_scale: float = 0.0,      # 0 → UFF 비활성
        uff_inner_steps: int = 8,
        uff_clamp: float = 1.0,
        uff_start_ratio: float = 0.0,         # (t/(T-1)) < ratio 이면 UFF/Vina skip
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        x0_fuse_corrector_weight: float = 0.5,# x0_fused = (1-w)*x0_pred + w*x0_corr
        # ---- Temperature 옵션 ----
        noise_temperature: float = 0.3,       # posterior 노이즈 온도
        # ---- UFF 동적 nonbonded 파라미터 ----
        uff_vdw_multiplier: float = 10.0,
        debug_log: bool = False,
        # ---- VinaSF 옵션 ----
        vina_guidance_scale: float = 0.0,     # 0 → Vina 비활성
        vina_inner_steps: int = 1,
        vina_clamp: float = 1.0,
        vina_perm_qb2rd=None,                 # (Nq,) permutation
    ):
        """
        Fixed Heun predictor-corrector + UFF/Vina Steering.

        CFG scheduling: cosine decay
        - t = T-1 (start, high noise): cfg(t) = cfg_scale
        - t = 0   (end,   low  noise): cfg(t) = cfg_floor

        Merge rule (v-space):
            v_hat = v_u + cfg(t) * (v_c - v_u)
        - cfg(t)=1    : pure conditional
        - cfg(t)=0    : pure unconditional
        - cfg(t)>1    : stronger-than-conditional CFG (가능)
        """

        import math  # 안전하게 내부 import (외부에 이미 있어도 OK)

        device = self.betas.device
        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- sanity for cfg params ----------
        cfg_scale = float(cfg_scale)
        cfg_floor = float(cfg_floor)
        if cfg_scale < 0.0:
            raise ValueError(f"cfg_scale must be >= 0, got {cfg_scale}")
        if cfg_floor < 0.0:
            raise ValueError(f"cfg_floor must be >= 0, got {cfg_floor}")

        # ---------- CFG cosine scheduling helper ----------
        # progress p: 0 at start (t=T-1) -> 1 at end (t=0)
        # cos_term: 1 -> 0
        # cfg(t) = cfg_floor + (cfg_scale - cfg_floor) * cos_term
        def _cfg_scale_cosine(step: int) -> float:
            if T <= 1:
                return cfg_scale
            denom = float(T - 1)
            p = float((T - 1) - step) / denom          # 0(start) -> 1(end)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0
            s = cfg_floor + (cfg_scale - cfg_floor) * cos_term
            # 수치오차 clamp
            if s < 0.0:
                s = 0.0
            return float(s)

        # ---------- 공통: query 노드 슬라이스 / 인덱스 ----------
        mol_slices_q = [
            (qb.batch == i).nonzero(as_tuple=True)[0]
            for i in range(qb.num_graphs)
        ]
        gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)  # [ΣNq]

        B = qb.num_graphs
        Nq = mol_slices_q[0].numel()
        for sl in mol_slices_q:
            assert sl.numel() == Nq, "현재 구현은 모든 query에 동일한 Nq를 가정합니다."

        # ---------- UFFTorch 세팅 ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs
            assert len(pocket_mols) == qb.num_graphs

            q_inputs = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            p_inputs = build_uff_inputs(
                pocket_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            qp_inputs = merge_uff_inputs(
                q_inputs,
                p_inputs,
                ignore_interfragment_interactions=False,
                vdw_distance_multiplier=float(uff_vdw_multiplier),
            )

            uff_model = UFFTorch(qp_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [
                        [conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                        for k in range(m.GetNumAtoms())
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols],
                dim=0,
            )  # [B,Np,3]
        else:
            uff_model = None
            pocket_coords_fixed = None

        # ---------- VinaSF 세팅 ----------
        use_vina = (vina_guidance_scale > 0.0)
        if use_vina:
            # NOTE: batch별 pocket/query가 다르면 여기 로직을 확장해야 함
            vina_model = VinaSFTorch.from_rdkit(pocket_mols[0], query_mols[0]).to(device).eval()
            if vina_perm_qb2rd is not None:
                vina_perm_qb2rd = vina_perm_qb2rd.to(device)

        # ---------- x_T 초기화 ----------
        x_t = torch.randn((qb.num_nodes, 3), device=device, dtype=torch.float32)

        # ---------- 메인 역과정 루프 ----------
        for t in reversed(range(T)):
            t_graph = torch.full((qb.num_graphs,), t, device=device, dtype=torch.long)

            # (NEW) timestep-dependent CFG scale (cosine: cfg_scale -> cfg_floor)
            cfg_t = _cfg_scale_cosine(t)

            # 1) Predictor: x_t -> v_hat -> x0_pred
            cur_q = qb.clone()
            cur_q.pos = x_t

            # --- CFG merge with schedule ---
            # cfg_t ≈ 1 -> pure conditional
            # cfg_t ≈ 0 -> pure unconditional
            if cfg_t <= 0.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=False)
            elif abs(cfg_t - 1.0) < 1e-8:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_t * (v_c - v_u)

            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]

            abar_t = self.sqrt_alphas_cumprod[t]
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t]

            x0_pred = abar_t * x_t - sig_t * v_hat

            # 2) Tentative Euler Step for Corrector: x0_pred -> x_{t-1}^{euler}
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_tm1_euler = c1_t * x0_pred + c2_t * x_t

            # 3) Corrector: x_{t-1}^{euler} -> v_hat_{t-1} -> x0_corr
            x0_corr = None
            if t > 0:
                t_graph2 = torch.full((qb.num_graphs,), t - 1, device=device, dtype=torch.long)

                # (NEW) use scheduled CFG at (t-1) for corrector pass too
                cfg_t2 = _cfg_scale_cosine(t - 1)

                cur_q2 = qb.clone()
                cur_q2.pos = x_tm1_euler

                if cfg_t2 <= 0.0:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=False)
                elif abs(cfg_t2 - 1.0) < 1e-8:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=True)
                else:
                    v_u2 = self(cur_q2, rb, t_graph2, condition=False)
                    v_c2 = self(cur_q2, rb, t_graph2, condition=True)
                    v_hat2_merged = v_u2 + cfg_t2 * (v_c2 - v_u2)

                merged2 = merge_graphs_in_batch(cur_q2, rb, device=device)
                qmask2 = (merged2.graph_idx % 2 == 0)
                v_hat2 = v_hat2_merged[qmask2]

                abar_t1 = self.sqrt_alphas_cumprod[t - 1]
                sig_t1  = self.sqrt_one_minus_alphas_cumprod[t - 1]

                x0_corr = abar_t1 * x_tm1_euler - sig_t1 * v_hat2

            # 4) Fusion: x0_fused
            if x0_corr is not None:
                w = float(x0_fuse_corrector_weight)
                x0_fused = (1.0 - w) * x0_pred + w * x0_corr
            else:
                x0_fused = x0_pred

            # 5) Physics Guidance (Steering on x0_fused)
            sigma_t = float(sig_t.item())
            gate_t = (max(0.0, min(1.0, 1.0 - sigma_t))) ** float(snr_gate_gamma)

            x0_star = x0_fused

            # 5-1) Vina Steering
            if use_vina and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0_v = x0_star.index_select(0, gather_idx_q)
                coords_q0_v = coords_q0_v.view(B, Nq, 3).contiguous().detach()
                coords_q_v = coords_q0_v.clone()

                inner_v = max(1, int(vina_inner_steps))
                for _ in range(inner_v):
                    coords_req = coords_q_v.clone()
                    coords_in = coords_req[:, vina_perm_qb2rd, :] if vina_perm_qb2rd is not None else coords_req

                    with torch.enable_grad():
                        score, grad = vina_model.score_and_gradient(coords_in)

                    if not isinstance(grad, torch.Tensor):
                        grad = torch.as_tensor(grad, device=device, dtype=coords_in.dtype)
                    if grad.ndim == 2:
                        grad = grad.unsqueeze(0)

                    if vina_perm_qb2rd is not None:
                        grad_q_v = torch.zeros_like(coords_req)
                        grad_q_v[:, vina_perm_qb2rd, :] = grad
                    else:
                        grad_q_v = grad

                    forces_v = (-grad_q_v).detach().clamp_(-vina_clamp, vina_clamp)
                    coords_q_v = (coords_q_v + vina_guidance_scale * forces_v).detach()

                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q_v.view(B * Nq, 3))

                if debug_log:
                    print(f"[Vina] t={t} cfg_t={cfg_t:.4f} ForceNorm={forces_v.norm(dim=-1).mean():.3f}")

            # 5-2) UFF Steering
            if use_uff and (t / max(1, T - 1)) >= uff_start_ratio and gate_t > 0.0:
                coords_q0 = x0_star.index_select(0, gather_idx_q)
                coords_q0 = coords_q0.view(B, Nq, 3).contiguous().detach()
                coords_0 = torch.cat([coords_q0, pocket_coords_fixed], dim=1)

                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_0)

                coords_q = coords_q0.clone()
                inner = max(1, int(uff_inner_steps))

                for _ in range(inner):
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)
                        coords_cat = torch.cat([coords_q_req, pocket_coords_fixed], dim=1)
                        E_b = uff_model(coords_cat)
                        if E_b.ndim == 0:
                            E_b = E_b.unsqueeze(0)
                        grad_q, = torch.autograd.grad(E_b.sum(), coords_q_req)

                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                    coords_q = (coords_q + uff_guidance_scale * forces_q).detach()

                x0_star = x0_star.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q.view(B * Nq, 3))

                if debug_log:
                    print(f"[UFF] t={t} cfg_t={cfg_t:.4f} ForceNorm={forces_q.norm(dim=-1).mean():.3f}")

            # 6) Final Transition: Compute μ_t using x0_star
            x_mean = c1_t * x0_star + c2_t * x_t

            # 7) Noise Addition
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)

                noise = torch.randn_like(x_t)
                x_t = x_mean + math.sqrt(var_t) * noise
            else:
                x_t = x_mean

        return (x_t, None)



    @torch.no_grad()
    def DDPM_Sampling_Heun_UFF_Vina_Post(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        clamp: float = 1e-10,
        cfg_scale: float = 1.0,
        # ---- UFF(PyTorch) 옵션 ----
        query_mols=None,                      # RDKit Mol 리스트 (len == num_graphs). None이면 UFF off
        pocket_mols=None,                     # RDKit Mol 리스트 (len == num_graphs). None이면 포켓 off
        uff_guidance_scale: float = 0.0,      # 0 → UFF 비활성
        uff_inner_steps: int = 8,             # UFF 내부 gradient step 횟수
        uff_clamp: float = 1.0,               # 힘 클램프
        uff_start_ratio: float = 0.0,         # (t/(T-1)) < ratio 이면 UFF/Vina skip
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        snr_use_zero_mode_mask: bool = False, # (isotropic 버전에서는 사용 안 함; 호환용)
        x0_fuse_corrector_weight: float = 0.5,# x0_fused = (1-w)*x0_pred + w*x0_corr
        # ---- Temperature 옵션 ----
        noise_temperature: float = 0.3,       # posterior 노이즈 온도 τ
        # ---- UFF 동적 nonbonded 파라미터 ----
        uff_vdw_multiplier: float = 10.0,     # 동적 컷오프 배수 (vdw_distance_multiplier)
        debug_log: bool = False,
        # ---- VinaSF 옵션 ----
        vina_guidance_scale: float = 0.0,     # 0 → Vina 비활성
        vina_inner_steps: int = 1,            # Vina 내부 gradient step 수
        vina_clamp: float = 1.0,              # Vina 힘 클램프
        vina_perm_qb2rd=None,                 # (Nq,) 그래프 heavy → Vina heavy perm, None이면 동일 순서
    ):
        """
        Heun predictor-corrector (v-param, isotropic) + UFFTorch + VinaSF steering.

        - 디퓨전: x-공간 v-param DDPM Heun (predictor-corrector).
        - UFF: torch-native UFFTorch로 batched 리간드-포켓 에너지/gradient 계산.
        * query_mols, pocket_mols → build_uff_inputs → UFFTorch(q_inputs)
        * 각 스텝에서 x0_fused 기준으로 리간드 좌표만 여러 번 업데이트.
        - VinaSF: VinaSFTorch.score_and_gradient로 리간드 포즈에 대한 score/gradient 받아서 추가 steering.
        - gate(t): σ_t = sqrt(1 - ᾱ_t), h(t) = 1 - σ_t, gate(t) = h(t)^γ
        → 후기 스텝에서 강하게 작동(late–strong).

        - Post optimization (추가):
        1) 최종 x_T에 대해 VinaSF local refinement (여러 step).
        2) 그 결과를 다시 UFFTorch로 여러 step 최소화해서 intra/inter 안정화.
        """

        device = self.betas.device

        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- 공통: query 노드 슬라이스 / 인덱스 ----------
        # 같은 토폴로지 + 동일 Nq 가정
        mol_slices_q = [
            (qb.batch == i).nonzero(as_tuple=True)[0]
            for i in range(qb.num_graphs)
        ]
        gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)  # [ΣNq]

        B = qb.num_graphs
        Nq = mol_slices_q[0].numel()
        for sl in mol_slices_q:
            assert sl.numel() == Nq, "현재 구현은 모든 query에 동일한 Nq를 가정합니다."

        # ---------- UFFTorch 세팅 ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs, (
                f"len(query_mols)={len(query_mols)} != qb.num_graphs={qb.num_graphs}"
            )
            assert len(pocket_mols) == qb.num_graphs, (
                f"len(pocket_mols)={len(pocket_mols)} != qb.num_graphs={qb.num_graphs}"
            )

            # RDKit Mol → UFFTorch 입력
            q_inputs = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            uff_model = UFFTorch(q_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            # pocket coords (고정)
            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [
                        [
                            conf.GetAtomPosition(k).x,
                            conf.GetAtomPosition(k).y,
                            conf.GetAtomPosition(k).z,
                        ]
                        for k in range(m.GetNumAtoms())
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols],
                dim=0,
            )  # [B,Np,3]
            Np = pocket_coords_fixed.shape[1]
        else:
            uff_model = None
            pocket_coords_fixed = None

        # ---------- VinaSF 세팅 ----------
        use_vina = (vina_guidance_scale > 0.0)
        if use_vina:
            # 단순히 첫 번째 pair로부터 VinaSF 객체 생성 (필요시 확장 가능)
            vina_model = VinaSFTorch.from_rdkit(pocket_mols[0], query_mols[0]).to(device).eval()
            if vina_perm_qb2rd is not None:
                vina_perm_qb2rd = vina_perm_qb2rd.to(device)

        # ---------- x_T 초기화 (표준 N(0,I)) ----------
        x_t = torch.randn(
            (qb.num_nodes, 3),
            device=device,
            dtype=torch.float32,
        )

        # ---------- 메인 역과정 루프 ----------
        for t in reversed(range(T)):
            t_graph = torch.full(
                (qb.num_graphs,),
                t,
                device=device,
                dtype=torch.long,
            )

            # 현재 좌표를 query batch에 세팅
            cur_q = qb.clone()
            cur_q.pos = x_t

            # v_hat@t (CFG)
            if cfg_scale == 1.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_scale * (v_c - v_u)

            # merge해서 query 노드만 선택
            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]  # [Nq_total,3] == qb.num_nodes

            # 계수 (스칼라)
            abar_t = self.sqrt_alphas_cumprod[t]           # √ᾱ_t
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t] # σ_t = √(1-ᾱ_t)

            # 1) x0_pred 복원 (v-param: x0 = √ᾱ x_t − √(1-ᾱ) v_t)
            x0_pred = abar_t * x_t - sig_t * v_hat

            # 2) posterior mean μ_t(x0_pred) → x_{t-1}^{euler}
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_tm1_euler = c1_t * x0_pred + c2_t * x_t

            # 3) corrector: t>0이면 v_hat@t-1 on x_{t-1}^{euler} → x0_corr
            if t > 0:
                t_graph2 = torch.full(
                    (qb.num_graphs,),
                    t - 1,
                    device=device,
                    dtype=torch.long,
                )
                cur_q2 = qb.clone()
                cur_q2.pos = x_tm1_euler

                if cfg_scale == 1.0:
                    v_hat2_merged = self(cur_q2, rb, t_graph2, condition=True)
                else:
                    v_u2 = self(cur_q2, rb, t_graph2, condition=False)
                    v_c2 = self(cur_q2, rb, t_graph2, condition=True)
                    v_hat2_merged = v_u2 + cfg_scale * (v_c2 - v_u2)

                merged2 = merge_graphs_in_batch(cur_q2, rb, device=device)
                qmask2 = (merged2.graph_idx % 2 == 0)
                v_hat2 = v_hat2_merged[qmask2]

                abar_t1 = self.sqrt_alphas_cumprod[t - 1]
                sig_t1  = self.sqrt_one_minus_alphas_cumprod[t - 1]

                x0_corr = abar_t1 * x_tm1_euler - sig_t1 * v_hat2
            else:
                x0_corr = None

            # 4) x0_fused = (1-w)*x0_pred + w*x0_corr
            if x0_corr is not None:
                w = float(x0_fuse_corrector_weight)
                x0_fused = (1.0 - w) * x0_pred + w * x0_corr
            else:
                x0_fused = x0_pred

            # 5) SNR-aware gate (h(t) = 1 - σ_t, gate = h^γ)
            sigma_t = float(sig_t.item())
            h_t = 1.0 - sigma_t
            h_t = max(0.0, min(1.0, h_t))
            gate_t = h_t ** float(snr_gate_gamma)

            # 6) 외부 에너지(UFF/Vina)로 x0_fused 보정 → x0_star
            x0_star = x0_fused

            # 6-1) UFF steering (역과정 중)
            if (
                use_uff
                and (t / max(1, T - 1)) >= uff_start_ratio
                and gate_t > 0.0
            ):
                # 현재 x0_fused에서 query 좌표만 [B,Nq,3]으로 reshape
                coords_q0 = x0_star.index_select(0, gather_idx_q)  # [ΣNq,3]
                coords_q0 = coords_q0.view(B, Nq, 3).detach()

                # nonbond candidate 업데이트는 grad 필요 없으니 no_grad 유지
                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_q0)

                coords_q = coords_q0.clone()

                inner = max(1, int(uff_inner_steps))
                # 여러 번 나눠서 업데이트
                step_scale = (uff_guidance_scale * gate_t) / float(inner)

                for _ in range(inner):
                    # 여기서만 grad tracking 켜기
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)  # [B,Nq,3]
                        coords_cat = torch.cat(
                            [coords_q_req, pocket_coords_fixed],
                            dim=1,
                        )  # [B,Nq+Np,3]

                        E_b = uff_model(coords_cat)  # (B,) or scalar

                        if not isinstance(E_b, torch.Tensor):
                            E_b = torch.as_tensor(E_b, device=device, dtype=coords_cat.dtype)
                        if E_b.ndim == 0:
                            E_b = E_b.unsqueeze(0)

                        grad_q, = torch.autograd.grad(
                            E_b.sum(),
                            coords_q_req,
                            create_graph=False,
                            retain_graph=False,
                        )

                    forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)  # [B,Nq,3]
                    coords_q = (coords_q + step_scale * forces_q).detach()

                # 최종 coords_q를 x0_star에 커밋
                x0_star = x0_star.clone()
                x0_star.index_copy_(
                    0,
                    gather_idx_q,
                    coords_q.reshape(B * Nq, 3),
                )

                if debug_log:
                    f_norm = forces_q.norm(dim=-1)
                    print(
                        f"[UFF] t={t:02d} gate={gate_t:.3f} | "
                        f"‖F‖ mean={f_norm.mean().item():.3f}, "
                        f"max={f_norm.max().item():.3f}"
                    )

            # 6-2) VinaSF steering (역과정 중, UFF 적용 이후 상태에서 추가 보정)
            if (
                use_vina
                and (t / max(1, T - 1)) >= uff_start_ratio
                and gate_t > 0.0
            ):
                coords_q0_v = x0_star.index_select(0, gather_idx_q)  # [ΣNq,3]
                coords_q0_v = coords_q0_v.view(B, Nq, 3).detach()
                coords_q_v = coords_q0_v.clone()

                inner_v = max(1, int(vina_inner_steps))
                step_v = (float(vina_guidance_scale) * gate_t) / float(inner_v)

                for _ in range(inner_v):
                    coords_req = coords_q_v.clone()  # [B,Nq,3]
                    if vina_perm_qb2rd is not None:
                        coords_in = coords_req[:, vina_perm_qb2rd, :]
                    else:
                        coords_in = coords_req

                    # VinaSFTorch는 score와 gradient를 직접 반환한다고 가정
                    with torch.enable_grad():
                        score, grad = vina_model.score_and_gradient(coords_in)

                    if not isinstance(grad, torch.Tensor):
                        grad = torch.as_tensor(
                            grad, device=device, dtype=coords_in.dtype
                        )
                    if grad.ndim == 2:
                        grad = grad.unsqueeze(0)  # [1,Nq,3] → [B,Nq,3] 형태 맞추기

                    if vina_perm_qb2rd is not None:
                        grad_q_v = torch.zeros_like(coords_req)
                        grad_q_v[:, vina_perm_qb2rd, :] = grad
                    else:
                        grad_q_v = grad

                    forces_v = (-grad_q_v).detach().clamp_(-vina_clamp, vina_clamp)
                    coords_q_v = (coords_q_v + step_v * forces_v).detach()

                # 최종 coords_q_v를 x0_star에 반영
                x0_star = x0_star.clone()
                x0_star.index_copy_(
                    0,
                    gather_idx_q,
                    coords_q_v.reshape(B * Nq, 3),
                )

                if debug_log:
                    f_v_norm = forces_v.norm(dim=-1)
                    print(
                        f"[Vina] t={t:02d} gate={gate_t:.3f} | "
                        f"‖F‖ mean={f_v_norm.mean().item():.3f}, "
                        f"max={f_v_norm.max().item():.3f}"
                    )

            # 7) μ_t(x0_star), μ_{t-1}(x0_star) → Heun 평균
            x_mu_t = c1_t * x0_star + c2_t * x_t

            if t > 0:
                c1_t1 = self.posterior_mean_coef1[t - 1]
                c2_t1 = self.posterior_mean_coef2[t - 1]
                x_mu_tm1 = c1_t1 * x0_star + c2_t1 * x_tm1_euler
                x_next_det = 0.5 * (x_mu_t + x_mu_tm1)
            else:
                x_next_det = x_mu_t

            # 8) posterior noise (+ temperature)
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)

                noise = torch.randn_like(x_t)
                x_t = x_next_det + math.sqrt(var_t) * noise
            else:
                x_t = x_next_det

        # ---------- Post optimization: joint Vina + UFF refinement ----------
        x_final = x_t

        # 하이퍼파라미터 (실험하면서 튜닝해봐)
        post_steps = 64                 # Vina+UFF 묶어서 몇 번 반복할지
        post_vina_step_scale = 0.02      # vina_guidance_scale와 곱해질 베이스 스텝 크기
        post_uff_step_scale = 0.1       # uff_guidance_scale와 곱해질 베이스 스텝 크기

        # 둘 중 하나라도 켜져있으면 joint refinement 진행
        if (use_vina or use_uff) and post_steps > 0:
            # 최종 좌표에서 query 부분만 [B, Nq, 3]으로 빼오기
            coords_q = x_final.index_select(0, gather_idx_q).view(B, Nq, 3).detach()

            # UFF nonbond candidate는 미리 한 번 업데이트
            if use_uff:
                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_q)

            for k in range(post_steps):
                # ----- 1) VinaSF local refinement step -----
                if use_vina:
                    coords_req = coords_q.clone()
                    if vina_perm_qb2rd is not None:
                        coords_in = coords_req[:, vina_perm_qb2rd, :]
                    else:
                        coords_in = coords_req

                    with torch.enable_grad():
                        score, grad = vina_model.score_and_gradient(coords_in)

                    if not isinstance(grad, torch.Tensor):
                        grad = torch.as_tensor(grad, device=device, dtype=coords_in.dtype)
                    if grad.ndim == 2:
                        grad = grad.unsqueeze(0)  # [1,Nq,3] → [B,Nq,3]

                    if vina_perm_qb2rd is not None:
                        grad_q = torch.zeros_like(coords_req)
                        grad_q[:, vina_perm_qb2rd, :] = grad
                    else:
                        grad_q = grad

                    forces_v = (-grad_q).detach().clamp_(-vina_clamp, vina_clamp)
                    step_v = post_vina_step_scale * float(vina_guidance_scale)
                    coords_q = (coords_q + step_v * forces_v).detach()

                    if debug_log and (k == 0 or (k + 1) == post_steps):
                        f_v_norm = forces_v.norm(dim=-1)
                        print(
                            f"[Post-Joint Vina] step={k+1}/{post_steps} | "
                            f"‖F‖ mean={f_v_norm.mean().item():.3f}, "
                            f"max={f_v_norm.max().item():.3f}"
                        )

                # ----- 2) UFFTorch refinement step -----
                if use_uff:
                    with torch.enable_grad():
                        coords_q_req = coords_q.clone().detach().requires_grad_(True)  # [B,Nq,3]
                        coords_cat = torch.cat(
                            [coords_q_req, pocket_coords_fixed],
                            dim=1,
                        )  # [B, Nq+Np, 3]

                        E_b = uff_model(coords_cat)
                        if not isinstance(E_b, torch.Tensor):
                            E_b = torch.as_tensor(E_b, device=device, dtype=coords_cat.dtype)
                        if E_b.ndim == 0:
                            E_b = E_b.unsqueeze(0)

                        grad_q, = torch.autograd.grad(
                            E_b.sum(),
                            coords_q_req,
                            create_graph=False,
                            retain_graph=False,
                        )

                    forces_u = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                    step_u = post_uff_step_scale * float(uff_guidance_scale)
                    coords_q = (coords_q + step_u * forces_u).detach()

                    if debug_log and (k == 0 or (k + 1) == post_steps):
                        f_u_norm = forces_u.norm(dim=-1)
                        print(
                            f"[Post-Joint UFF] step={k+1}/{post_steps} | "
                            f"‖F‖ mean={f_u_norm.mean().item():.3f}, "
                            f"max={f_u_norm.max().item():.3f}"
                        )

            # joint refinement 결과를 x_final에 반영
            x_final = x_final.clone()
            x_final.index_copy_(0, gather_idx_q, coords_q.reshape(B * Nq, 3))

        return (x_final, None)