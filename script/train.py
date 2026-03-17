# train_diffalign_single_gpu.py
# - 스케줄 버퍼(betas/alphas_*)만 드랍하고 안전하게 체크포인트 로드
# - 단일 GPU 학습 + CosineAnnealingWarmRestarts
# - reference가 없는 샘플도 안전 처리

import os
import re
import argparse
import copy
import traceback
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch_geometric.data import Data, Batch
from tqdm import tqdm
import numpy as np

# ==== 프로젝트 의존 경로 ====
from models.epsnet.diffusion import DiffAlign  # 모델 경로 확인
from utils.datasets import ConformationDataset  # 데이터셋 경로 확인
# 필요시 transforms, misc 사용
# from utils.transforms import *
# from utils.misc import *


# ------------------------------------------------------------
# 1) 체크포인트에서 '스케줄 버퍼'만 제거하는 도우미
# ------------------------------------------------------------
DROP_SUFFIXES = (
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "alphas",
    "sqrt_alphas",
    "posterior_variance",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
)

def _extract_state_dict(ckpt_obj):
    """Lightning/일반/커스텀 케이스에서 state_dict만 뽑아오기."""
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "module", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj if isinstance(ckpt_obj, dict) else ckpt_obj

def drop_diffusion_buffers(state_dict, verbose=True):
    """등록 버퍼(스케줄 텐서)만 제거. 모듈 prefix(DataParallel 등) 포함한 키도 안전 처리."""
    keys = list(state_dict.keys())
    to_drop = []
    for k in keys:
        # 끝 토큰 기준
        if k.split(".")[-1] in DROP_SUFFIXES:
            to_drop.append(k)
            continue
        # 혹시 모듈 중간 경로 포함 방어
        for suf in DROP_SUFFIXES:
            # 키의 끝이 .suf 로 끝나는 패턴 커버
            if re.search(rf"(?:^|\.){re.escape(suf)}$", k):
                to_drop.append(k)
                break
    for k in to_drop:
        state_dict.pop(k, None)
    if verbose:
        print(f"[drop_diffusion_buffers] dropped {len(to_drop)} keys:")
        for k in to_drop:
            print("  -", k)
    return state_dict

def load_checkpoint_safely(model, ckpt_path, strict=False, verbose=True):
    """체크포인트를 로드하되, diffusion 스케줄 버퍼만 제거하고 가중치만 주입."""
    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        if verbose and ckpt_path is not None:
            print(f"[load_checkpoint_safely] checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        print("[load_checkpoint_safely] invalid checkpoint format; skip loading.")
        return

    sd = drop_diffusion_buffers(sd, verbose=verbose)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=strict)

    if verbose:
        print("[load_state_dict] strict =", strict)
        if missing_keys:
            print("  missing:", missing_keys)
        if unexpected_keys:
            print("  unexpected:", unexpected_keys)
    print(f"[load_checkpoint_safely] Successfully loaded (sans schedule buffers) from: {ckpt_path}")


# ------------------------------------------------------------
# 2) 데이터 콜레이트: query/reference + 고정 배치 노이즈
# ------------------------------------------------------------
def custom_collate(data_list):
    """
    데이터 리스트를 받아 query와 reference 배치를 생성하고 노이즈를 추가하는 collate 함수.
    - 배치 공통 3D 노이즈를 pos에 더함
    - reference가 없으면 None으로 반환
    """
    query_data_list = []
    ref_data_list = []

    # 배치 전체에 동일하게 적용할 3D 노이즈 (CPU에서 생성)
    noise = torch.randn(3) * 0.1

    for data in data_list:
        # Query
        atom_type = getattr(data, 'atom_type', None)
        pos = getattr(data, 'pos', None)
        edge_index = getattr(data, 'edge_index', None)
        edge_type = getattr(data, 'edge_type', None)

        if atom_type is None:
            # 원자 타입이 없으면 스킵
            continue

        if pos is None:
            pos = torch.zeros((atom_type.shape[0], 3), dtype=torch.float32)

        q = Data(
            atom_type=atom_type,
            edge_index=edge_index,
            edge_type=edge_type,
            pos=pos + noise,  # 공통 노이즈
        )
        query_data_list.append(q)

        # Reference (있으면)
        ref_pos = getattr(data, 'pos_r', None)
        ref_atom_type = getattr(data, 'atom_type_r', None)
        ref_edge_index = getattr(data, 'edge_index_r', None)
        ref_edge_type = getattr(data, 'edge_type_r', None)

        if (ref_pos is not None) and (ref_atom_type is not None):
            r = Data(
                atom_type=ref_atom_type,
                edge_index=ref_edge_index,
                edge_type=ref_edge_type,
                pos=ref_pos + noise,  # 동일 노이즈
            )
            ref_data_list.append(r)
        else:
            ref_data_list.append(None)

    if not query_data_list:
        return None, None

    query_batch = Batch.from_data_list(query_data_list)

    valid_ref = [r for r in ref_data_list if r is not None]
    reference_batch = Batch.from_data_list(valid_ref) if valid_ref else None

    return query_batch, reference_batch


# ------------------------------------------------------------
# 3) 학습 루프(단일 GPU, CosineWarmRestarts per-step 업데이트)
# ------------------------------------------------------------
def ensure_reference_batch(reference_batch, device):
    """
    reference_batch가 None일 때도 모델이 안전하게 돌 수 있도록
    빈 Batch를 생성해 반환 (필요 시 merge에서 사용 가능).
    """
    if reference_batch is not None:
        return reference_batch
    # 빈 그래프 Batch
    empty = Batch()
    # torch_geometric Batch는 속성이 없어도 동작하지만,
    # .to(device) 호출을 위해 그냥 반환
    return empty.to(device)

def train(args):
    print("Running training on a single GPU.")

    # --- 장치/기본 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
    torch.set_printoptions(precision=2, sci_mode=False)

    # --- 데이터셋 ---
    data_path = 'data/extended_remove_h_centered_v2.pkl'  # 경로 확인
    train_set = ConformationDataset(data_path, transform=None)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=custom_collate,
    )
    print(f"Data loaded. Dataset size: {len(train_set)}, DataLoader batch size: {args.batch_size}")

    # --- 모델 ---
    # ※ 여기를 4스텝로 맞추고 싶으면 DiffAlign(num_timesteps=4, ...) 로 생성
    model = DiffAlign(num_timesteps=args.num_timesteps).to(device)
    print(device)
    print(f"Model initialized on {device} (num_timesteps={args.num_timesteps}).")

    # --- 체크포인트 로드 (스케줄 버퍼 드랍) ---
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        load_checkpoint_safely(model, args.checkpoint, strict=False, verbose=True)
    else:
        if args.checkpoint:
            print(f"[warn] checkpoint not found: {args.checkpoint} → training from scratch")
        else:
            print("No checkpoint specified → training from scratch")

    # --- 옵티마이저 & 스케줄러 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=args.t_mult,
        eta_min=args.eta_min,
    )

    global_step = 0
    model.train()
    print("Starting training loop (Cosine Warm Restarts)...")

    for it in range(args.iter_num):
        losses = []
        epoch_loss_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"[Iter {it+1}/{args.iter_num}]", leave=True)

        for step_idx, batch in pbar:
            if batch is None or batch[0] is None:
                continue

            query_batch, reference_batch = batch

            # device로 이동
            query_batch = query_batch.to(device, non_blocking=True)
            reference_batch = ensure_reference_batch(reference_batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss = model.get_loss(
                query_batch=query_batch,
                reference_batch=reference_batch,
            )

            # NaN/Inf 방지
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"\n[warn] NaN/Inf loss at iter {it+1}, step {global_step}. Skip.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # === 스케줄러 per-step 업데이트 (분수 에폭) ===
            global_step += 1
            epoch_progress = (step_idx / len(train_loader)) if len(train_loader) > 0 else 0.0
            scheduler.step(it + epoch_progress)

            # 로깅
            loss_item = float(loss.detach().item())
            losses.append(loss_item)
            epoch_loss_sum += loss_item
            epoch_steps += 1

            current_lr = optimizer.param_groups[0]['lr']
            running_avg_loss = epoch_loss_sum / max(1, epoch_steps)
            pbar.set_postfix({
                'loss': f'{loss_item:.4f}',
                'avg_loss': f'{running_avg_loss:.4f}',
                'lr': f"{current_lr:.2e}",
            })

        # --- 이터 종료 로그 ---
        if losses:
            loss_mean = float(np.mean(losses))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Iter {it+1}/{args.iter_num}] Avg Loss: {loss_mean:.4f}, LR: {current_lr:.2e}")
        else:
            print(f"[Iter {it+1}/{args.iter_num}] No valid loss recorded.")

        # --- 체크포인트 저장 ---
        if (it + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'{it+1}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"[ckpt] saved to {save_path}")

    print("Training finished.")


# ------------------------------------------------------------
# 4) 엔트리포인트
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffAlign Single GPU Training (safe load, cosine restarts)')
    parser.add_argument('--iter_num', type=int, default=1000, help='Number of training iterations (epochs)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial(max) learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--save_dir', type=str, default='./param', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=1, help='Save every N iters')
    parser.add_argument('--checkpoint', type=str, default='./param/64.pt', help='(Optional) checkpoint path to load')
    parser.add_argument('--num_timesteps', type=int, default=32, help='Target T for this run (e.g., 4)')
    # Cosine Warm Restarts
    parser.add_argument('--t0', type=int, default=64, help='First cycle length (epochs)')
    parser.add_argument('--t_mult', type=int, default=1, help='Cycle length multiplier')
    parser.add_argument('--eta_min', type=float, default=5e-7, help='Minimum LR')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train(args)

