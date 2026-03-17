#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate disco diffalign runs for a single tag group.

This script discovers run files from:
  disco/<TARGET>/PDB_Structures/<QUERY>[_320_wo_scheduling][_<TAG>]_<SEED>.sdf

Ground-truth ligand for RMSD is:
  disco/<TARGET>/PDB_Structures/<QUERY>_ChI.sdf

Outputs:
- selected_runs.csv
- run_metrics.csv
- errors.csv
- success_rate_1A_2A_3A.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from rdkit import Chem

from eval_dataset_rmsd_tanimoto import (
    VinaSFTorch,
    calc_vina_scores_batch,
    heavy_rmsd,
    load_all_mols,
    load_first_mol,
    safe_prob,
    write_csv,
)


RunKey = Tuple[str, str, int]
RUN_RE = re.compile(r"^([0-9A-Za-z]{4})(?:_320_wo_scheduling)?(?:_(.+))?_([0-9]+)\.sdf$")
DUD_E_CODES = {
    "3BI6", "2UZ3", "3WFF", "1LI4", "1Q4X", "3CCT", "3K23", "4HLW", "2X7D", "3HL5",
    "3D4S", "1DB5", "2W8Y", "1X78", "3NF8", "1BSK", "2XWE", "3OAP", "1BL4", "3ZPQ",
    "2RD6", "4LY1", "1C1B", "1SYI", "2RL5", "3VYD", "1NFU", "3M2W", "2ZNK", "3PYY",
    "3KK6", "2ON6", "2QE4", "1YP9", "3D5F", "1B9T", "3FVN", "2Z7G", "3SFF", "2OC2",
    "4J52", "1K7L", "3BEA", "3OW4", "3TWJ", "1NJS", "1BNU", "2OG8", "1V4S", "1W0Y",
    "2X7O", "2R2W", "3LGP", "4C61", "4FC0", "3ZCL", "3RLP", "2ZEC", "3GCQ", "3U9W",
    "3G0F", "2ZMM", "3E8D", "3TVC", "3I60", "2IIT", "3BZ3", "2I4U", "4OTY", "3ZM4",
    "2ZDU", "3I81", "4D89", "2O7N", "1D3G", "1ZGB", "3B2R", "3TZ7", "1R9O", "1S3B",
    "1C8K", "1G74", "3IKA", "3NXO", "1BID", "4EMA", "3QQK", "2F94", "1QBQ", "3GSG",
    "3OE8", "2CNK", "2HV5", "3QAK", "3NXU",
}
SELECTED_RUN_FIELDS = [
    "target",
    "query_code",
    "seed",
    "method",
    "tag",
    "sdf_path",
]
RUN_METRIC_FIELDS = [
    "target",
    "query_code",
    "seed",
    "method",
    "tag",
    "rmsd",
    "lt_1A",
    "lt_2A",
    "lt_3A",
    "sdf_path",
]
ERROR_FIELDS = [
    "target",
    "query_code",
    "seed",
    "method",
    "tag",
    "sdf_path",
    "reason",
]
SUCCESS_FIELDS = [
    "method",
    "tag_filter",
    "threshold_A",
    "n_runs",
    "success_n",
    "success_prob",
]


def parse_seeds(text: str) -> List[int]:
    seeds = set()
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        if "-" in tok:
            a_str, b_str = tok.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            lo, hi = (a, b) if a <= b else (b, a)
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(tok))
    if not seeds:
        raise ValueError("Empty seed set.")
    return sorted(seeds)


def discover_runs(disco_root: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for target_dir in sorted(disco_root.iterdir()):
        if not target_dir.is_dir():
            continue
        struct_dir = target_dir / "PDB_Structures"
        if not struct_dir.is_dir():
            continue

        for sdf in sorted(struct_dir.glob("*.sdf")):
            m = RUN_RE.match(sdf.name)
            if not m:
                continue
            query_code, tag_raw, seed_str = m.groups()
            runs.append(
                {
                    "target": target_dir.name,
                    "query_code": query_code,
                    "tag": tag_raw or "",
                    "seed": int(seed_str),
                    "sdf_path": sdf,
                }
            )
    return runs


def run_key(run: Dict[str, object]) -> RunKey:
    return (
        str(run["target"]),
        str(run["query_code"]),
        int(run["seed"]),
    )


def tag_matches(
    tag: str,
    *,
    exact: str,
    keyword: str,
    untagged: bool,
) -> bool:
    if untagged:
        return tag == ""
    if exact:
        return tag == exact
    if keyword:
        return keyword.lower() in tag.lower()
    return False


def build_tag_filter_text(args: argparse.Namespace) -> str:
    if bool(args.untagged):
        return "untagged"
    if args.tag_exact.strip():
        return f"exact:{args.tag_exact.strip()}"
    if args.tag_keyword.strip():
        return f"keyword:{args.tag_keyword.strip()}"
    return "unknown"


def resolve_method_name(args: argparse.Namespace) -> str:
    if args.method_name.strip():
        return args.method_name.strip()
    if bool(args.untagged):
        return "untagged"
    if args.tag_exact.strip():
        return args.tag_exact.strip()
    if args.tag_keyword.strip():
        return args.tag_keyword.strip()
    return "diffalign"


def pick_unique_by_key(
    runs: Sequence[Dict[str, object]],
    *,
    method_name: str,
) -> Tuple[List[Dict[str, object]], int]:
    out: Dict[RunKey, Dict[str, object]] = {}
    dups = 0
    for r in sorted(runs, key=lambda x: str(x["sdf_path"])):
        k = run_key(r)
        if k in out:
            dups += 1
            continue
        rr = dict(r)
        rr["method"] = method_name
        out[k] = rr
    return [out[k] for k in sorted(out.keys())], dups


def select_runs(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], int, int, int]:
    all_runs = discover_runs(args.disco_root)
    if not all_runs:
        raise RuntimeError(f"No disco runs found under {args.disco_root}")

    seeds = set(parse_seeds(args.seeds))
    candidates: List[Dict[str, object]] = []
    for run in all_runs:
        if int(run["seed"]) not in seeds:
            continue
        if not tag_matches(
            str(run["tag"]),
            exact=args.tag_exact.strip(),
            keyword=args.tag_keyword.strip(),
            untagged=bool(args.untagged),
        ):
            continue
        candidates.append(run)

    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]

    selected_runs, duplicates = pick_unique_by_key(
        candidates,
        method_name=resolve_method_name(args),
    )
    return selected_runs, duplicates, len(all_runs), len(candidates)


def resolve_template_code(disco_root: Path, target: str) -> str | None:
    struct_dir = disco_root / target / "PDB_Structures"
    if not struct_dir.is_dir():
        return None
    candidates = sorted(
        {
            p.name[:4]
            for p in struct_dir.glob("*_ChI.sdf")
            if len(p.name) >= 4 and p.name[:4] in DUD_E_CODES
        }
    )
    if not candidates:
        return None
    return candidates[0]


def pick_top_pose(
    *,
    sdf_path: Path,
    selector: str,
    max_poses: int,
    receptor_mol: object | None = None,
    device: torch.device | None = None,
) -> object | None:
    mols = load_all_mols(
        sdf_path,
        sanitize=True,
        remove_hs=True,
        max_poses=max_poses,
    )
    if not mols:
        return None
    if len(mols) == 1 or selector == "first":
        return mols[0]

    if selector == "vina":
        if receptor_mol is None:
            return mols[0]
        use_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scores = calc_vina_scores_batch(receptor_mol, mols, device=use_device)
        if not scores:
            return mols[0]
        best_idx = min(range(len(scores)), key=lambda i: float(scores[i]))
        return mols[best_idx]

    if selector == "uff":
        best = None
        best_e = None
        for mol in mols:
            if not mol.HasProp("UFF_Energy"):
                continue
            try:
                energy = float(mol.GetProp("UFF_Energy"))
            except Exception:
                continue
            if best_e is None or energy < best_e:
                best_e = energy
                best = mol
        return best if best is not None else mols[0]

    raise ValueError(f"Unknown selector: {selector}")


def evaluate_selected_runs(
    runs: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    gt_cache: Dict[Tuple[str, str], object | None] = {}
    template_code_cache: Dict[str, str | None] = {}
    receptor_cache: Dict[str, object | None] = {}
    vina_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pose_selector == "vina":
        print(f"[INFO] vina selector device={vina_device}")

    for i, run in enumerate(runs, start=1):
        target = str(run["target"])
        query_code = str(run["query_code"])
        seed = int(run["seed"])
        tag = str(run["tag"])
        method = str(run["method"])
        sdf_path = Path(run["sdf_path"])

        if i % 500 == 0:
            print(f"[INFO] progress {i}/{len(runs)}")

        gt_key = (target, query_code)
        if gt_key not in gt_cache:
            gt_path = args.disco_root / target / "PDB_Structures" / f"{query_code}_ChI.sdf"
            gt_cache[gt_key] = load_first_mol(gt_path, sanitize=True, remove_hs=True)
        gt_mol = gt_cache[gt_key]
        if gt_mol is None:
            error_rows.append(
                {
                    "target": target,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": "gt_load_fail_query_chi",
                }
            )
            continue

        try:
            receptor_mol = None
            if args.pose_selector == "vina":
                if target not in template_code_cache:
                    template_code_cache[target] = resolve_template_code(args.disco_root, target)
                template_code = template_code_cache[target]
                if not template_code:
                    raise RuntimeError("template_code_not_found")
                if target not in receptor_cache:
                    rec_path = args.disco_root / target / "PDB_Structures" / f"{template_code}_POC.pdb"
                    receptor_cache[target] = Chem.MolFromPDBFile(str(rec_path), removeHs=True)
                receptor_mol = receptor_cache[target]
                if receptor_mol is None:
                    raise RuntimeError("receptor_load_fail")

            pred_mol = pick_top_pose(
                sdf_path=sdf_path,
                selector=args.pose_selector,
                max_poses=args.max_poses,
                receptor_mol=receptor_mol,
                device=vina_device,
            )
            if pred_mol is None:
                raise RuntimeError("no_valid_pose")

            rmsd = heavy_rmsd(pred_mol, gt_mol)
            if rmsd is None:
                raise RuntimeError("rmsd_fail")
        except Exception as exc:
            error_rows.append(
                {
                    "target": target,
                    "query_code": query_code,
                    "seed": seed,
                    "method": method,
                    "tag": tag,
                    "sdf_path": str(sdf_path),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        rmsd_value = float(rmsd)
        run_rows.append(
            {
                "target": target,
                "query_code": query_code,
                "seed": seed,
                "method": method,
                "tag": tag,
                "rmsd": f"{rmsd_value:.6f}",
                "lt_1A": int(rmsd_value < 1.0),
                "lt_2A": int(rmsd_value < 2.0),
                "lt_3A": int(rmsd_value < 3.0),
                "sdf_path": str(sdf_path),
            }
        )

    return run_rows, error_rows


def build_success_rows(
    run_rows: Sequence[Dict[str, object]],
    *,
    method_name: str,
    tag_filter: str,
) -> List[Dict[str, object]]:
    n_runs = len(run_rows)
    out: List[Dict[str, object]] = []
    for threshold in (1.0, 2.0, 3.0):
        success_n = sum(1 for row in run_rows if float(row["rmsd"]) < threshold)
        out.append(
            {
                "method": method_name,
                "tag_filter": tag_filter,
                "threshold_A": f"{threshold:.1f}",
                "n_runs": n_runs,
                "success_n": success_n,
                "success_prob": f"{safe_prob(success_n, n_runs):.6f}",
            }
        )
    return out


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_outputs(
    *,
    out_dir: Path,
    selected_rows: Sequence[Dict[str, object]],
    run_rows: Sequence[Dict[str, object]],
    error_rows: Sequence[Dict[str, object]],
    success_rows: Sequence[Dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "selected_runs.csv",
        list(selected_rows),
        fieldnames=SELECTED_RUN_FIELDS,
    )
    write_csv(
        out_dir / "run_metrics.csv",
        list(run_rows),
        fieldnames=RUN_METRIC_FIELDS,
    )
    write_csv(
        out_dir / "errors.csv",
        list(error_rows),
        fieldnames=ERROR_FIELDS,
    )
    write_csv(
        out_dir / "success_rate_1A_2A_3A.csv",
        list(success_rows),
        fieldnames=SUCCESS_FIELDS,
    )


def print_summary(
    *,
    success_rows: Sequence[Dict[str, object]],
    selected_count: int,
    valid_count: int,
    error_count: int,
    duplicates: int,
) -> None:
    print("\n===== DISCO DIFFALIGN EVAL =====")
    print(
        f"selected_runs={selected_count} valid_runs={valid_count} "
        f"errors={error_count} duplicates_ignored={duplicates}"
    )
    for row in success_rows:
        print(
            f"RMSD<{row['threshold_A']}A: "
            f"{row['success_prob']} ({row['success_n']}/{row['n_runs']})"
        )


def run_single_or_shard(args: argparse.Namespace) -> int:
    try:
        selected_runs, duplicates, total_runs, filtered_count = select_runs(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(
        f"[INFO] discovered runs total={total_runs} filtered={filtered_count} "
        f"method={resolve_method_name(args)} tag_filter={build_tag_filter_text(args)}"
    )
    print(f"[INFO] pose_selector={args.pose_selector} max_poses={args.max_poses}")

    if args.mode == "shard":
        total_selected = len(selected_runs)
        selected_runs = [
            run
            for idx, run in enumerate(selected_runs)
            if (idx % args.num_shards) == args.shard_index
        ]
        print(
            f"[INFO] mode=shard shard={args.shard_index}/{args.num_shards} "
            f"runs_in_shard={len(selected_runs)} total_selected={total_selected}"
        )

    if not selected_runs:
        if args.mode == "shard":
            write_outputs(
                out_dir=args.out_dir,
                selected_rows=[],
                run_rows=[],
                error_rows=[],
                success_rows=build_success_rows(
                    [],
                    method_name=resolve_method_name(args),
                    tag_filter=build_tag_filter_text(args),
                ),
            )
            print("[INFO] Empty shard after filtering. Wrote empty outputs.")
            return 0
        print("[ERROR] No runs matched the requested tag filter and seed set.")
        return 1

    selected_rows = [
        {
            "target": str(run["target"]),
            "query_code": str(run["query_code"]),
            "seed": int(run["seed"]),
            "method": str(run["method"]),
            "tag": str(run["tag"]),
            "sdf_path": str(run["sdf_path"]),
        }
        for run in selected_runs
    ]
    run_rows, error_rows = evaluate_selected_runs(selected_runs, args)
    success_rows = build_success_rows(
        run_rows,
        method_name=resolve_method_name(args),
        tag_filter=build_tag_filter_text(args),
    )
    write_outputs(
        out_dir=args.out_dir,
        selected_rows=selected_rows,
        run_rows=run_rows,
        error_rows=error_rows,
        success_rows=success_rows,
    )

    if not run_rows:
        if args.mode == "shard":
            print("[INFO] Shard has no valid runs after RMSD evaluation. Wrote empty summaries.")
            return 0
        print("[ERROR] No valid runs after RMSD evaluation.")
        return 1

    print_summary(
        success_rows=success_rows,
        selected_count=len(selected_rows),
        valid_count=len(run_rows),
        error_count=len(error_rows),
        duplicates=duplicates,
    )
    print(f"success_csv: {args.out_dir / 'success_rate_1A_2A_3A.csv'}")
    return 0


def run_merge(args: argparse.Namespace) -> int:
    shard_root = args.shards_root if args.shards_root is not None else args.out_dir
    shard_dirs = sorted([path for path in shard_root.glob(args.shard_dir_glob) if path.is_dir()])
    if not shard_dirs:
        print(f"[ERROR] No shard directories found: root={shard_root} glob={args.shard_dir_glob}")
        return 1

    selected_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for shard_dir in shard_dirs:
        selected_rows.extend(load_csv_rows(shard_dir / "selected_runs.csv"))
        run_rows.extend(load_csv_rows(shard_dir / "run_metrics.csv"))
        error_rows.extend(load_csv_rows(shard_dir / "errors.csv"))

    if not selected_rows and not run_rows and not error_rows:
        print("[ERROR] No shard outputs to merge.")
        return 1

    method_name = resolve_method_name(args)
    if run_rows:
        method_name = str(run_rows[0].get("method", method_name))

    success_rows = build_success_rows(
        run_rows,
        method_name=method_name,
        tag_filter=build_tag_filter_text(args),
    )

    def sort_key(row: Dict[str, object]) -> Tuple[str, str, int]:
        return (
            str(row.get("target", "")),
            str(row.get("query_code", "")),
            int(row.get("seed", 0)),
        )

    selected_rows = sorted(selected_rows, key=sort_key)
    run_rows = sorted(run_rows, key=sort_key)
    error_rows = sorted(
        error_rows,
        key=lambda row: (sort_key(row), str(row.get("reason", ""))),
    )

    write_outputs(
        out_dir=args.out_dir,
        selected_rows=selected_rows,
        run_rows=run_rows,
        error_rows=error_rows,
        success_rows=success_rows,
    )

    if not run_rows:
        print("[ERROR] No valid shard run_metrics.csv rows to merge.")
        return 1

    print("\n===== DISCO DIFFALIGN MERGED EVAL =====")
    print(
        f"shards={len(shard_dirs)} selected_runs={len(selected_rows)} "
        f"valid_runs={len(run_rows)} errors={len(error_rows)}"
    )
    for row in success_rows:
        print(
            f"RMSD<{row['threshold_A']}A: "
            f"{row['success_prob']} ({row['success_n']}/{row['n_runs']})"
        )
    print(f"success_csv: {args.out_dir / 'success_rate_1A_2A_3A.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate disco diffalign RMSD success @ 1A/2A/3A for one tag group."
    )
    parser.add_argument("--mode", choices=["single", "shard", "merge"], default="single")
    parser.add_argument("--disco-root", type=Path, default=Path("./disco"))
    parser.add_argument("--seeds", type=str, default="1-10")
    parser.add_argument("--tag", "--tag-exact", dest="tag_exact", type=str, default="")
    parser.add_argument("--tag-keyword", type=str, default="")
    parser.add_argument("--untagged", action="store_true")
    parser.add_argument("--method-name", type=str, default="")
    parser.add_argument("--pose-selector", choices=["first", "uff", "vina"], default="vina")
    parser.add_argument("--max-poses", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--shards-root",
        type=Path,
        default=None,
        help="Root directory containing shard outputs (merge mode).",
    )
    parser.add_argument(
        "--shard-dir-glob",
        type=str,
        default="shard_*",
        help="Glob pattern for shard directories under --shards-root.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/disco_diffalign_eval"))
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.max_poses <= 0:
        print("[ERROR] --max-poses must be >= 1.")
        return 1

    if args.mode == "merge":
        return run_merge(args)

    if not args.disco_root.exists():
        print(f"[ERROR] disco root not found: {args.disco_root}")
        return 1

    if args.mode == "shard":
        if args.num_shards < 1:
            print("[ERROR] --num-shards must be >= 1 in shard mode.")
            return 1
        if not (0 <= args.shard_index < args.num_shards):
            print("[ERROR] shard-index must satisfy 0 <= shard-index < num-shards.")
            return 1

    if args.pose_selector == "vina" and VinaSFTorch is None:
        print("[ERROR] --pose-selector=vina requires vinasf_torch, but it is not available.")
        return 1

    tag_mode_count = (
        int(bool(args.untagged))
        + int(bool(args.tag_exact.strip()))
        + int(bool(args.tag_keyword.strip()))
    )
    if tag_mode_count != 1:
        print("[ERROR] Exactly one of --tag/--tag-keyword/--untagged must be set.")
        return 1

    return run_single_or_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
