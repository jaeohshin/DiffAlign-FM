#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset diffalign evaluator.

Metrics:
1) Success probability for RMSD < 1A, <2A, <3A
2) 2D Tanimoto (template-query) binned by 0.1 and
   success probability P(RMSD < 2A) in each bin.

Modes:
- single: evaluate everything in one process
- shard: evaluate one shard (for multi-GPU parallel runs)
- merge: merge shard CSVs and compute final summaries
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

try:
    from vinasf_torch import VinaSF_PyTorch
except Exception:
    try:
        from vinasf_torch import VinaSFTorch as VinaSF_PyTorch
    except Exception:
        VinaSF_PyTorch = None

# Backward-compatible alias for other modules importing this symbol.
VinaSFTorch = VinaSF_PyTorch


RDLogger.DisableLog("rdApp.*")
OBRMS_BIN = os.environ.get("OBRMS_BIN", "obrms")

# <cluster>_<template>_<query>_diffalign_cartesian_320_uff_vina_pocket_<tag>_<seed>.sdf
FILE_RE = re.compile(
    r"^([^_]+)_([0-9A-Za-z]{4})_([0-9A-Za-z]{4})_diffalign_cartesian_320_uff_vina_pocket_(.+)_([0-9]+)\.sdf$"
)


def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("Empty seed list.")
    return sorted(set(out))


def parse_tags(text: str) -> List[str]:
    out: List[str] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return sorted(set(out))


def safe_prob(num: int, den: int) -> float:
    return (num / den) if den > 0 else 0.0


def load_first_mol(path: Path, *, sanitize: bool = True, remove_hs: bool = True) -> Optional[Chem.Mol]:
    if not path.exists():
        return None
    suppl = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
    for mol in suppl:
        if mol is not None:
            return mol
    return None


def load_all_mols(
    path: Path,
    *,
    sanitize: bool,
    remove_hs: bool,
    max_poses: int,
) -> List[Chem.Mol]:
    if not path.exists():
        return []
    suppl = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
    out: List[Chem.Mol] = []
    for mol in suppl:
        if mol is None:
            continue
        out.append(mol)
        if len(out) >= max_poses:
            break
    return out


def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(a, b) -> float:
    return float(DataStructs.TanimotoSimilarity(a, b))


def _obrms_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "pred.sdf"
        f2 = Path(tmpdir) / "ref.sdf"
        Chem.MolToMolFile(mol1, str(f1))
        Chem.MolToMolFile(mol2, str(f2))
        proc = subprocess.run(
            [OBRMS_BIN, str(f1), str(f2)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out = proc.stdout.strip()
        for tok in out.replace("RMSD", " ").replace("rmsd", " ").split():
            try:
                return float(tok)
            except ValueError:
                continue
        raise RuntimeError(f"Failed to parse obrms output: {out}")


def heavy_rmsd(pred_mol: Chem.Mol, gt_mol: Chem.Mol) -> Optional[float]:
    def _n_heavy(m: Chem.Mol) -> int:
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)

    def _remove_hs_relaxed(m: Chem.Mol) -> Optional[Chem.Mol]:
        mc = Chem.Mol(m)
        try:
            return Chem.RemoveHs(mc)
        except Exception:
            pass
        try:
            return Chem.RemoveHs(mc, sanitize=False)
        except Exception:
            return None

    try:
        pred = Chem.Mol(pred_mol)
        ref = Chem.Mol(gt_mol)
        if pred.GetNumConformers() == 0 or ref.GetNumConformers() == 0:
            return None
        if _n_heavy(pred) != _n_heavy(ref):
            return None

        pred_noh = _remove_hs_relaxed(pred)
        ref_noh = _remove_hs_relaxed(ref)
        if (
            pred_noh is not None
            and ref_noh is not None
            and pred_noh.GetNumConformers() > 0
            and ref_noh.GetNumConformers() > 0
            and pred_noh.GetNumAtoms() == ref_noh.GetNumAtoms()
            and pred_noh.GetNumAtoms() > 0
        ):
            return _obrms_rmsd(pred_noh, ref_noh)
        return _obrms_rmsd(pred, ref)
    except Exception:
        return None


def calc_vina_scores_batch(
    receptor_mol: Chem.Mol,
    pose_mols: List[Chem.Mol],
    device: torch.device,
) -> List[float]:
    if not pose_mols:
        return []
    if VinaSF_PyTorch is None:
        raise RuntimeError("vinasf_torch is not available")

    vina = VinaSF_PyTorch.from_rdkit(receptor_mol, pose_mols[0]).to(device)
    ref_coords = vina.ligand.pose_heavy_atoms_coords
    dtype = ref_coords.dtype
    n_heavy = ref_coords.shape[-2]

    coords_list = []
    for mol in pose_mols:
        mol_noh = Chem.RemoveHs(mol)
        if mol_noh.GetNumAtoms() != n_heavy:
            raise RuntimeError(
                f"Heavy atom count mismatch (expected={n_heavy}, got={mol_noh.GetNumAtoms()})"
            )
        conf = mol_noh.GetConformer()
        pos = torch.tensor(conf.GetPositions(), device=device, dtype=dtype)
        coords_list.append(pos)

    coords_batch = torch.stack(coords_list, dim=0)
    with torch.enable_grad():
        scores, _ = vina.score_and_gradient(coords_batch)
    return scores.view(-1).detach().cpu().tolist()


def pick_top_pose(
    sdf_path: Path,
    receptor_mol: Optional[Chem.Mol],
    device: torch.device,
    selector: str,
    max_poses: int,
) -> Optional[Chem.Mol]:
    mols = load_all_mols(
        sdf_path,
        sanitize=True,
        remove_hs=True,
        max_poses=max_poses,
    )
    if not mols:
        return None
    if len(mols) == 1:
        return mols[0]
    if selector == "first":
        return mols[0]

    if selector == "uff":
        best = None
        best_e = None
        for m in mols:
            if not m.HasProp("UFF_Energy"):
                continue
            try:
                e = float(m.GetProp("UFF_Energy"))
            except Exception:
                continue
            if best_e is None or e < best_e:
                best_e = e
                best = m
        return best if best is not None else mols[0]

    if selector == "vina":
        if receptor_mol is None:
            return mols[0]
        scores = calc_vina_scores_batch(receptor_mol, mols, device=device)
        if not scores:
            return mols[0]
        idx = int(np.argmin(np.asarray(scores, dtype=np.float32)))
        return mols[idx]

    raise ValueError(f"Unknown selector: {selector}")


def map_method(tag: str) -> str:
    if tag == "0.05":
        return "ours"
    if tag == "0_no_vina":
        return "ours_no_vina"
    if tag == "0_no_vina_no_uff":
        return "ours_no_vina_no_uff"
    return tag


def discover_runs(dataset_root: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for sdf in sorted(dataset_root.glob("*/*/diffalign/*.sdf")):
        m = FILE_RE.match(sdf.name)
        if not m:
            continue
        cluster, template_code, query_code, tag, seed_text = m.groups()
        runs.append(
            {
                "cluster": cluster,
                "template_code": template_code,
                "query_code": query_code,
                "tag": tag,
                "method": map_method(tag),
                "seed": int(seed_text),
                "sdf_path": sdf,
            }
        )
    return runs


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def shard_tag(shard_index: int, num_shards: int) -> str:
    return f"shard_{shard_index:02d}_of_{num_shards:02d}"


def summarize_rows(run_rows: List[Dict[str, object]], bin_width: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n_bins = int(round(1.0 / bin_width))
    succ_counts = defaultdict(lambda: {"n": 0, "lt1": 0, "lt2": 0, "lt3": 0})
    bin_counts = defaultdict(lambda: [0] * n_bins)
    bin_succ2 = defaultdict(lambda: [0] * n_bins)

    for r in run_rows:
        method = str(r["method"])
        tani = float(r["tanimoto"])
        rmsd = float(r["rmsd"])

        stat = succ_counts[method]
        stat["n"] += 1
        if rmsd < 1.0:
            stat["lt1"] += 1
        if rmsd < 2.0:
            stat["lt2"] += 1
        if rmsd < 3.0:
            stat["lt3"] += 1

        t_clip = max(0.0, min(tani, 1.0 - 1e-8))
        b = int(t_clip / bin_width)
        b = max(0, min(n_bins - 1, b))
        bin_counts[method][b] += 1
        if rmsd < 2.0:
            bin_succ2[method][b] += 1

    methods = sorted(succ_counts.keys())
    summary_rows: List[Dict[str, object]] = []
    for method in methods:
        s = succ_counts[method]
        n = int(s["n"])
        summary_rows.append(
            {
                "method": method,
                "n_runs": n,
                "succ_lt_3A_n": int(s["lt3"]),
                "succ_lt_3A_prob": f"{safe_prob(int(s['lt3']), n):.6f}",
                "succ_lt_2A_n": int(s["lt2"]),
                "succ_lt_2A_prob": f"{safe_prob(int(s['lt2']), n):.6f}",
                "succ_lt_1A_n": int(s["lt1"]),
                "succ_lt_1A_prob": f"{safe_prob(int(s['lt1']), n):.6f}",
            }
        )

    bin_rows: List[Dict[str, object]] = []
    for method in methods:
        for i in range(n_bins):
            lo = i * bin_width
            hi = min(1.0, lo + bin_width)
            cnt = int(bin_counts[method][i])
            succ2 = int(bin_succ2[method][i])
            bin_rows.append(
                {
                    "method": method,
                    "bin_lo": f"{lo:.3f}",
                    "bin_hi": f"{hi:.3f}",
                    "pair_count": cnt,
                    "succ_rmsd_lt_2A_n": succ2,
                    "succ_rmsd_lt_2A_prob": f"{safe_prob(succ2, cnt):.6f}",
                }
            )
    return summary_rows, bin_rows


def evaluate_runs(
    runs: List[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.top1_selector == "vina":
        print(f"[INFO] vina selector device={device}")

    gt_cache: Dict[Tuple[str, str], Optional[Chem.Mol]] = {}
    template_cache: Dict[Tuple[str, str], Optional[Chem.Mol]] = {}
    receptor_cache: Dict[Tuple[str, str], Optional[Chem.Mol]] = {}
    tani_cache: Dict[Tuple[str, str, str], Optional[float]] = {}

    run_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for i, r in enumerate(runs, start=1):
        cluster = str(r["cluster"])
        template_code = str(r["template_code"])
        query_code = str(r["query_code"])
        method = str(r["method"])
        tag = str(r["tag"])
        seed = int(r["seed"])
        sdf_path: Path = r["sdf_path"]  # type: ignore[assignment]

        if i % 500 == 0:
            print(f"[INFO] progress {i}/{len(runs)}")

        gt_key = (cluster, query_code)
        if gt_key not in gt_cache:
            gt_path = args.dataset_root / cluster / query_code / f"{query_code}_prot" / f"{query_code}_l.sdf"
            gt_cache[gt_key] = load_first_mol(gt_path, sanitize=True, remove_hs=True)
        gt_mol = gt_cache[gt_key]
        if gt_mol is None:
            error_rows.append(
                {
                    "cluster": cluster,
                    "template_code": template_code,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": "gt_load_fail",
                }
            )
            continue

        tmpl_key = (cluster, template_code)
        if tmpl_key not in template_cache:
            tmpl_path = (
                args.dataset_root / cluster / template_code / f"{template_code}_prot" / f"{template_code}_l.sdf"
            )
            template_cache[tmpl_key] = load_first_mol(tmpl_path, sanitize=True, remove_hs=True)
        template_mol = template_cache[tmpl_key]
        if template_mol is None:
            error_rows.append(
                {
                    "cluster": cluster,
                    "template_code": template_code,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": "template_load_fail",
                }
            )
            continue

        tani_key = (cluster, template_code, query_code)
        if tani_key not in tani_cache:
            try:
                tani_cache[tani_key] = tanimoto(morgan_fp(template_mol), morgan_fp(gt_mol))
            except Exception:
                tani_cache[tani_key] = None
        tani = tani_cache[tani_key]
        if tani is None or (not math.isfinite(float(tani))):
            error_rows.append(
                {
                    "cluster": cluster,
                    "template_code": template_code,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": "tanimoto_fail",
                }
            )
            continue

        receptor_mol: Optional[Chem.Mol] = None
        if args.top1_selector == "vina":
            if tmpl_key not in receptor_cache:
                rec_path = (
                    args.dataset_root / cluster / template_code / f"{template_code}_prot" / f"{template_code}_p.pdb"
                )
                receptor_cache[tmpl_key] = Chem.MolFromPDBFile(str(rec_path), removeHs=True)
            receptor_mol = receptor_cache[tmpl_key]

        try:
            pred_mol = pick_top_pose(
                sdf_path=sdf_path,
                receptor_mol=receptor_mol,
                device=device,
                selector=args.top1_selector,
                max_poses=args.max_poses,
            )
            if pred_mol is None:
                raise RuntimeError("no_valid_pose")
            rmsd = heavy_rmsd(pred_mol, gt_mol)
            if rmsd is None or not math.isfinite(float(rmsd)):
                raise RuntimeError("rmsd_fail")
        except Exception as exc:
            error_rows.append(
                {
                    "cluster": cluster,
                    "template_code": template_code,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        run_rows.append(
            {
                "cluster": cluster,
                "template_code": template_code,
                "query_code": query_code,
                "seed": seed,
                "method": method,
                "tag": tag,
                "tanimoto": f"{float(tani):.6f}",
                "rmsd": f"{float(rmsd):.6f}",
                "sdf_path": str(sdf_path),
            }
        )

    return run_rows, error_rows


def prepare_filtered_runs(args: argparse.Namespace) -> List[Dict[str, object]]:
    all_runs = discover_runs(args.dataset_root)
    if not all_runs:
        raise RuntimeError("No diffalign runs found under dataset.")

    seeds = set(parse_seeds(args.seeds))
    tags = set(parse_tags(args.tags)) if args.tags.strip() else set()

    filtered: List[Dict[str, object]] = []
    for r in all_runs:
        if r["seed"] not in seeds:
            continue
        if tags and r["tag"] not in tags:
            continue
        filtered.append(r)
    if args.max_runs > 0:
        filtered = filtered[: args.max_runs]
    if not filtered:
        raise RuntimeError("No runs after filtering by seeds/tags.")
    print(f"[INFO] discovered runs total={len(all_runs)} filtered={len(filtered)}")
    return filtered


def run_single_or_shard(args: argparse.Namespace) -> int:
    if args.mode == "shard":
        if args.num_shards < 1:
            print("[ERROR] --num-shards must be >= 1 in shard mode.")
            return 1
        if not (0 <= args.shard_index < args.num_shards):
            print("[ERROR] shard-index must satisfy 0 <= shard-index < num-shards.")
            return 1

    try:
        filtered = prepare_filtered_runs(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.mode == "shard":
        filtered = [r for idx, r in enumerate(filtered) if (idx % args.num_shards) == args.shard_index]
        print(
            f"[INFO] mode=shard shard={args.shard_index}/{args.num_shards} "
            f"runs_in_shard={len(filtered)}"
        )
        if not filtered:
            print("[INFO] Empty shard. Writing empty outputs.")

    print(f"[INFO] top1_selector={args.top1_selector} max_poses={args.max_poses}")
    run_rows, error_rows = evaluate_runs(filtered, args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        run_csv = out_dir / "run_metrics.csv"
        error_csv = out_dir / "errors.csv"
        summary_csv = out_dir / "method_summary.csv"
        bins_csv = out_dir / "tanimoto_bins_rmsd2.csv"
    else:
        tag = shard_tag(args.shard_index, args.num_shards)
        run_csv = out_dir / f"run_metrics_{tag}.csv"
        error_csv = out_dir / f"errors_{tag}.csv"
        summary_csv = out_dir / f"method_summary_{tag}.csv"
        bins_csv = out_dir / f"tanimoto_bins_rmsd2_{tag}.csv"

    write_csv(
        run_csv,
        run_rows,
        fieldnames=[
            "cluster",
            "template_code",
            "query_code",
            "seed",
            "method",
            "tag",
            "tanimoto",
            "rmsd",
            "sdf_path",
        ],
    )
    write_csv(
        error_csv,
        error_rows,
        fieldnames=[
            "cluster",
            "template_code",
            "query_code",
            "seed",
            "method",
            "tag",
            "sdf_path",
            "reason",
        ],
    )

    summary_rows, bin_rows = summarize_rows(run_rows, args.bin_width)
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "method",
            "n_runs",
            "succ_lt_3A_n",
            "succ_lt_3A_prob",
            "succ_lt_2A_n",
            "succ_lt_2A_prob",
            "succ_lt_1A_n",
            "succ_lt_1A_prob",
        ],
    )
    write_csv(
        bins_csv,
        bin_rows,
        fieldnames=[
            "method",
            "bin_lo",
            "bin_hi",
            "pair_count",
            "succ_rmsd_lt_2A_n",
            "succ_rmsd_lt_2A_prob",
        ],
    )

    print("\n===== METHOD SUMMARY =====")
    for r in summary_rows:
        print(
            f"{r['method']}: n={r['n_runs']} "
            f"P(RMSD<3A)={r['succ_lt_3A_prob']} "
            f"P(RMSD<2A)={r['succ_lt_2A_prob']} "
            f"P(RMSD<1A)={r['succ_lt_1A_prob']}"
        )

    print("\n===== OUTPUT =====")
    print(f"run_metrics: {run_csv}")
    print(f"errors:      {error_csv}")
    print(f"summary:     {summary_csv}")
    print(f"bins:        {bins_csv}")
    print(f"valid_runs={len(run_rows)} errors={len(error_rows)}")
    return 0


def run_merge(args: argparse.Namespace) -> int:
    if args.num_shards < 1:
        print("[ERROR] --num-shards must be >= 1 in merge mode.")
        return 1

    merged_runs: List[Dict[str, object]] = []
    merged_errors: List[Dict[str, object]] = []
    missing_shards = 0

    for i in range(args.num_shards):
        tag = shard_tag(i, args.num_shards)
        run_csv = args.out_dir / f"run_metrics_{tag}.csv"
        err_csv = args.out_dir / f"errors_{tag}.csv"

        run_rows_raw = load_csv(run_csv)
        err_rows_raw = load_csv(err_csv)

        if not run_rows_raw and not err_rows_raw:
            missing_shards += 1
            continue

        for r in run_rows_raw:
            merged_runs.append(
                {
                    "cluster": r.get("cluster", ""),
                    "template_code": r.get("template_code", ""),
                    "query_code": r.get("query_code", ""),
                    "seed": int(r.get("seed", "0") or "0"),
                    "method": r.get("method", ""),
                    "tag": r.get("tag", ""),
                    "tanimoto": float(r.get("tanimoto", "nan")),
                    "rmsd": float(r.get("rmsd", "nan")),
                    "sdf_path": r.get("sdf_path", ""),
                }
            )
        for r in err_rows_raw:
            merged_errors.append({k: v for k, v in r.items()})

    if missing_shards > 0:
        print(f"[WARN] missing shard files: {missing_shards}/{args.num_shards}")

    if not merged_runs and not merged_errors:
        print("[ERROR] No shard outputs to merge.")
        return 1

    summary_rows, bin_rows = summarize_rows(merged_runs, args.bin_width)

    run_csv = args.out_dir / "run_metrics.csv"
    err_csv = args.out_dir / "errors.csv"
    summary_csv = args.out_dir / "method_summary.csv"
    bins_csv = args.out_dir / "tanimoto_bins_rmsd2.csv"

    write_csv(
        run_csv,
        [
            {
                "cluster": r["cluster"],
                "template_code": r["template_code"],
                "query_code": r["query_code"],
                "seed": r["seed"],
                "method": r["method"],
                "tag": r["tag"],
                "tanimoto": f"{float(r['tanimoto']):.6f}",
                "rmsd": f"{float(r['rmsd']):.6f}",
                "sdf_path": r["sdf_path"],
            }
            for r in merged_runs
        ],
        fieldnames=[
            "cluster",
            "template_code",
            "query_code",
            "seed",
            "method",
            "tag",
            "tanimoto",
            "rmsd",
            "sdf_path",
        ],
    )
    write_csv(
        err_csv,
        merged_errors,
        fieldnames=[
            "cluster",
            "template_code",
            "query_code",
            "seed",
            "method",
            "tag",
            "sdf_path",
            "reason",
        ],
    )
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "method",
            "n_runs",
            "succ_lt_3A_n",
            "succ_lt_3A_prob",
            "succ_lt_2A_n",
            "succ_lt_2A_prob",
            "succ_lt_1A_n",
            "succ_lt_1A_prob",
        ],
    )
    write_csv(
        bins_csv,
        bin_rows,
        fieldnames=[
            "method",
            "bin_lo",
            "bin_hi",
            "pair_count",
            "succ_rmsd_lt_2A_n",
            "succ_rmsd_lt_2A_prob",
        ],
    )

    print("\n===== MERGED METHOD SUMMARY =====")
    for r in summary_rows:
        print(
            f"{r['method']}: n={r['n_runs']} "
            f"P(RMSD<3A)={r['succ_lt_3A_prob']} "
            f"P(RMSD<2A)={r['succ_lt_2A_prob']} "
            f"P(RMSD<1A)={r['succ_lt_1A_prob']}"
        )

    print("\n===== OUTPUT =====")
    print(f"run_metrics: {run_csv}")
    print(f"errors:      {err_csv}")
    print(f"summary:     {summary_csv}")
    print(f"bins:        {bins_csv}")
    print(f"valid_runs={len(merged_runs)} errors={len(merged_errors)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate dataset diffalign RMSD success and Tanimoto-binned success."
    )
    p.add_argument("--mode", choices=["single", "shard", "merge"], default="single")
    p.add_argument("--dataset-root", type=Path, default=Path("./dataset"))
    p.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument(
        "--tags",
        type=str,
        default="",
        help="Optional comma-separated tags, e.g. 0.05,0_no_vina,0_no_vina_no_uff",
    )
    p.add_argument("--top1-selector", choices=["first", "uff", "vina"], default="vina")
    p.add_argument("--max-poses", type=int, default=20)
    p.add_argument("--bin-width", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=Path("analysis/dataset_eval"))
    p.add_argument("--max-runs", type=int, default=-1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.dataset_root.exists():
        print(f"[ERROR] dataset root not found: {args.dataset_root}")
        return 1
    if args.bin_width <= 0.0 or args.bin_width > 1.0:
        print("[ERROR] --bin-width must be in (0,1].")
        return 1
    if args.max_poses <= 0:
        print("[ERROR] --max-poses must be >= 1.")
        return 1
    if args.mode in ("single", "shard") and args.top1_selector == "vina" and VinaSF_PyTorch is None:
        print("[ERROR] --top1-selector=vina requires vinasf_torch, but it is not available.")
        return 1

    if args.mode == "merge":
        return run_merge(args)
    return run_single_or_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
