#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from torch_geometric.data import Data, Batch

from models.epsnet.diffusion import DiffAlign
from utils.chem import *  # set_rdmol_positions 등

# ===== RDKit 로그 끄기 =====
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def load_first_mol_from_sdf(sdf_path: str, *, sanitize: bool = True, removeHs: bool = True):
    """
    SDF에서 첫 번째로 파싱되는 mol을 반환.
    (sanitize=True이면 읽어오는 순간 RDKit sanitization 수행)
    """
    if not os.path.exists(sdf_path):
        return None

    suppl = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=removeHs)
    for m in suppl:
        if m is not None:
            return m
    return None


def force_remove_all_hydrogens(mol):
    """
    Remove all hydrogen atoms (atomic number == 1), including explicit H atoms that
    may remain after RDKit supplier removeHs=True.
    """
    if mol is None:
        return None

    rw_mol = Chem.RWMol(Chem.Mol(mol))
    hydrogen_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 1]
    if not hydrogen_indices:
        return Chem.Mol(mol)

    for idx in reversed(hydrogen_indices):
        rw_mol.RemoveAtom(idx)

    stripped = rw_mol.GetMol()
    # Best-effort sanitize after atom deletion (do not fail hard on partial issues).
    Chem.SanitizeMol(stripped, catchErrors=True)
    return stripped


def bond_type_to_int(bond) -> int:
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return 12
    else:
        print(f"Warning: Unknown bond type {bond_type}. Assigning default (1).")
        return 1


def mol_to_graph_data_obj(mol):
    if mol is None:
        return None

    try:
        # Node features (atomic numbers)
        atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.long)

        # Edge features (bond types) and connectivity
        edges = []
        bond_types = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bond_type_to_int(bond)

            edges.extend([(i, j), (j, i)])
            bond_types.extend([bt, bt])

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(bond_types, dtype=torch.long)

        # Atom positions
        if mol.GetNumConformers() == 0:
            print("Warning: No conformer found in mol_to_graph_data_obj")
            return None

        conf = mol.GetConformer(0)
        coordinates = conf.GetPositions()
        pos = torch.tensor(coordinates, dtype=torch.float)

        data = Data(atom_type=x, edge_index=edge_index, edge_type=edge_attr, pos=pos)
        data.num_nodes = mol.GetNumAtoms()
        return data

    except Exception as e:
        print(f"Error converting molecule to graph: {e}")
        return None


def log_sampling_failure(log_path: str, folder_name: str, query_code: str, attempt: int, seed: int, exc: Exception):
    tb = traceback.format_exc()
    msg = (
        f"\n[{datetime.now().isoformat()}] FAIL {folder_name}/{query_code} "
        f"attempt={attempt} seed={seed} exc={type(exc).__name__}: {exc}\n{tb}\n"
    )
    print(msg)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass


def is_bad_pos_tensor(x) -> bool:
    """pos_gen 방어: None/빈 텐서/NaN/inf 포함이면 True"""
    if x is None:
        return True
    if (not torch.is_tensor(x)) or x.numel() == 0:
        return True
    if not torch.isfinite(x).all():
        return True
    return False


def make_vina_tag(vina_guidance_scale: float) -> str:
    if abs(vina_guidance_scale) < 1e-12:
        return "no_vina"
    scale_str = f"{vina_guidance_scale:g}".replace(".", "p").replace("-", "m")
    return f"vina_{scale_str}"


def make_uff_tag(uff_guidance_scale: float) -> str:
    if abs(uff_guidance_scale) < 1e-12:
        return "no_uff"
    scale_str = f"{uff_guidance_scale:g}".replace(".", "p").replace("-", "m")
    return f"uff_{scale_str}"


def build_output_tag(base_tag: str, use_self_condition: bool, *, tag_is_final: bool = False) -> str:
    tag = base_tag.strip().replace(" ", "_")
    if tag_is_final:
        return tag
    if not use_self_condition:
        return tag
    if ("self-condition" in tag) or ("self_condition" in tag):
        return tag
    if not tag:
        return "self-condition"
    return f"{tag}_self-condition"


def _checkpoint_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def resolve_checkpoint_path(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    param_dir = requested_path.parent
    candidates = sorted(param_dir.glob("*.pt"), key=_checkpoint_sort_key)
    if not candidates:
        raise FileNotFoundError(
            f"Missing model checkpoint: {requested_path} "
            f"(and no fallback *.pt found in {param_dir})"
        )

    fallback = candidates[-1]
    print(
        f"[WARN] Requested checkpoint not found: {requested_path}. "
        f"Using fallback checkpoint: {fallback}"
    )
    return fallback


def load_manifest_jobs(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"job manifest not found: {manifest_path}")

    jobs = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                idx = int(row.get("global_query_index", "").strip())
            except Exception as exc:
                raise ValueError(f"Invalid global_query_index in manifest row: {row}") from exc

            job = {
                "global_query_index": idx,
                "folder_name": row.get("folder_name", "").strip(),
                "struct_dir": row.get("struct_dir", "").strip(),
                "template_name": row.get("template_name", "").strip(),
                "query_code": row.get("query_code", "").strip(),
                "template_sdf": row.get("template_sdf", "").strip(),
                "pocket_pdb": row.get("pocket_pdb", "").strip(),
                "query_sdf": row.get("query_sdf", "").strip(),
            }
            required = ("folder_name", "struct_dir", "template_name", "query_code", "template_sdf", "pocket_pdb", "query_sdf")
            missing = [k for k in required if not job[k]]
            if missing:
                raise ValueError(f"Manifest row missing fields {missing}: {row}")
            jobs.append(job)

    jobs.sort(key=lambda x: int(x["global_query_index"]))
    return jobs


def discover_jobs_from_disco(folder_names, dude_set):
    jobs = []
    global_idx = 0
    for folder_name in folder_names:
        struct_dir = f"./disco/{folder_name}/PDB_Structures"
        if not os.path.isdir(struct_dir):
            continue

        file_names = sorted([fn for fn in os.listdir(struct_dir) if fn.endswith("_ChI.sdf")])
        if not file_names:
            continue

        template_candidates = sorted({fn[:4] for fn in file_names if fn[:4] in dude_set})
        if not template_candidates:
            continue
        template_name = template_candidates[0]

        template_sdf_path = os.path.join(struct_dir, f"{template_name}_ChI.sdf")
        pocket_path = os.path.join(struct_dir, f"{template_name}_POC.pdb")

        for file_name in file_names:
            query_code = file_name[:4]
            if query_code == template_name:
                continue

            jobs.append(
                {
                    "global_query_index": global_idx,
                    "folder_name": folder_name,
                    "struct_dir": struct_dir,
                    "template_name": template_name,
                    "query_code": query_code,
                    "template_sdf": template_sdf_path,
                    "pocket_pdb": pocket_path,
                    "query_sdf": os.path.join(struct_dir, file_name),
                }
            )
            global_idx += 1
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./param/320.pt",
        help="DiffAlign checkpoint path. Falls back to latest ./param/*.pt if missing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducible sampling")
    parser.add_argument(
        "--vina-guidance-scale",
        type=float,
        default=0.0,
        help="Vina guidance scale. 0.0 disables Vina guidance.",
    )
    parser.add_argument(
        "--uff-guidance-scale",
        type=float,
        default=0.05,
        help="UFF guidance scale used during sampling. Default: 0.05",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help=(
            "Optional filename tag appended to distinguish outputs without overwriting "
            "existing SDF files."
        ),
    )
    parser.add_argument(
        "--output-tag-is-final",
        action="store_true",
        help="Use --output-tag as the final tag without appending self-condition suffixes.",
    )
    parser.add_argument(
        "--use-self-condition",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable self-conditioning in diffusion sampling.",
    )
    parser.add_argument(
        "--self-condition-use-steered-x0",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use guidance-steered x0 as the next self-conditioning input.",
    )
    parser.add_argument(
        "--self-condition-source",
        type=str,
        default=None,
        choices=["x0_pred", "x_star"],
        help=(
            "Optional alias for the self-conditioning input source. "
            "x0_pred reuses predictor output, x_star reuses guided x0_star and "
            "overrides --self-condition-use-steered-x0."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split query jobs into this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Optional limit for selected jobs in this shard. 0 means no limit.",
    )
    parser.add_argument(
        "--job-manifest",
        type=str,
        default="",
        help="Optional CSV manifest path with precomputed jobs to avoid repeated disco scanning.",
    )
    parser.add_argument(
        "--job-stride",
        type=int,
        default=0,
        help="Optional strided job assignment on global_query_index. >0 enables stride filtering.",
    )
    parser.add_argument(
        "--job-offset",
        type=int,
        default=-1,
        help="Offset in [0, job_stride) used with --job-stride.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist and are non-empty.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="./sampling_failures.log",
        help="Path for sampling failure logs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of sampled poses per query.",
    )
    parser.add_argument(
        "--uff-inner-steps",
        type=int,
        default=4,
        help="Number of inner optimization steps for UFF guidance.",
    )
    parser.add_argument(
        "--vina-inner-steps",
        type=int,
        default=4,
        help="Number of inner optimization steps for Vina guidance.",
    )
    parser.add_argument(
        "--compute-uff-energy",
        type=int,
        default=1,
        choices=[0, 1],
        help="Compute per-pose UFF_Energy before writing SDF. Disable for faster sampling throughput.",
    )
    parser.add_argument(
        "--sanitize-output",
        type=int,
        default=1,
        choices=[0, 1],
        help="Run RDKit sanitize before writing each generated pose.",
    )
    parser.add_argument(
        "--guidance-mol-copy",
        type=str,
        default="deep",
        choices=["shallow", "deep"],
        help=(
            "shallow: reuse one RDKit mol object per query in guidance lists (faster). "
            "deep: deep-copy mol objects for each sample and attempt (legacy-safe, slower)."
        ),
    )
    parser.add_argument(
        "--rdkit-io-mode",
        type=str,
        default="optimized",
        choices=["optimized", "legacy"],
        help=(
            "optimized: build per-pose RDKit mol once and reuse for energy+write; "
            "legacy: build separately for energy and writer (original behavior)."
        ),
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable deterministic torch/cuDNN setup for reproducible outputs across runs.",
    )
    parser.add_argument(
        "--use-harmonic-prior",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use harmonic prior for x_T initialization.",
    )
    parser.add_argument(
        "--harmonic-spring-k",
        type=float,
        default=10.0,
        help="Spring constant used by harmonic prior.",
    )
    parser.add_argument(
        "--harmonic-langevin-steps",
        type=int,
        default=50,
        help="Langevin steps used by harmonic prior.",
    )
    args = parser.parse_args()

    if bool(args.deterministic):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ===== seed 고정 =====
    base_seed = int(args.seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    vina_guidance_scale = float(args.vina_guidance_scale)
    if vina_guidance_scale < 0.0:
        raise ValueError("--vina-guidance-scale must be >= 0.0")
    uff_guidance_scale = float(args.uff_guidance_scale)
    if uff_guidance_scale < 0.0:
        raise ValueError("--uff-guidance-scale must be >= 0.0")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.max_jobs < 0:
        raise ValueError("--max-jobs must be >= 0")
    if args.job_stride < 0:
        raise ValueError("--job-stride must be >= 0")
    if args.job_offset < -1:
        raise ValueError("--job-offset must be >= -1")
    if args.job_stride > 0:
        if args.job_offset < 0 or args.job_offset >= args.job_stride:
            raise ValueError("--job-offset must satisfy 0 <= job-offset < job-stride when job-stride > 0")
    elif args.job_offset != -1:
        raise ValueError("--job-offset requires --job-stride > 0")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be >= 1")
    if args.uff_inner_steps <= 0:
        raise ValueError("--uff-inner-steps must be >= 1")
    if args.vina_inner_steps <= 0:
        raise ValueError("--vina-inner-steps must be >= 1")
    if args.harmonic_spring_k <= 0.0:
        raise ValueError("--harmonic-spring-k must be > 0")
    if args.harmonic_langevin_steps <= 0:
        raise ValueError("--harmonic-langevin-steps must be >= 1")
    use_self_condition = bool(args.use_self_condition)
    if not use_self_condition:
        self_condition_source = "disabled"
    elif args.self_condition_source is not None:
        self_condition_source = str(args.self_condition_source)
    elif bool(args.self_condition_use_steered_x0):
        self_condition_source = "x_star"
    else:
        self_condition_source = "x0_pred"
    self_condition_use_steered_x0 = (self_condition_source == "x_star")
    use_harmonic_prior = bool(args.use_harmonic_prior)
    harmonic_spring_k = float(args.harmonic_spring_k)
    harmonic_langevin_steps = int(args.harmonic_langevin_steps)
    num_samples = int(args.num_samples)
    uff_inner_steps = int(args.uff_inner_steps)
    vina_inner_steps = int(args.vina_inner_steps)
    compute_uff_energy = bool(args.compute_uff_energy)
    sanitize_output = bool(args.sanitize_output)
    vina_enabled = (vina_guidance_scale > 0.0)
    vina_tag = make_vina_tag(vina_guidance_scale)
    uff_tag = make_uff_tag(uff_guidance_scale)
    extra_tag = build_output_tag(
        args.output_tag,
        use_self_condition=use_self_condition,
        tag_is_final=bool(args.output_tag_is_final),
    )

    # Keep legacy naming by default, but add UFF tag when non-default to avoid collisions.
    run_tag_parts = [vina_tag]
    if abs(uff_guidance_scale - 0.05) > 1e-12:
        run_tag_parts.append(uff_tag)
    if extra_tag:
        run_tag_parts.append(extra_tag)
    run_tag = "_".join(run_tag_parts)

    print(f"[INFO] vina_guidance_scale={vina_guidance_scale:.6f} ({vina_tag})")
    print(f"[INFO] uff_guidance_scale={uff_guidance_scale:.6f} ({uff_tag})")
    print(f"[INFO] use_self_condition={use_self_condition}")
    print(f"[INFO] self_condition_source={self_condition_source}")
    print(f"[INFO] self_condition_use_steered_x0={self_condition_use_steered_x0}")
    print(f"[INFO] output_tag_is_final={bool(args.output_tag_is_final)}")
    print(f"[INFO] num_shards={args.num_shards}")
    print(f"[INFO] shard_index={args.shard_index}")
    print(f"[INFO] job_manifest={args.job_manifest if args.job_manifest else '(none)'}")
    print(f"[INFO] job_stride={args.job_stride}")
    print(f"[INFO] job_offset={args.job_offset}")
    print(f"[INFO] max_jobs={args.max_jobs}")
    print(f"[INFO] skip_existing={bool(args.skip_existing)}")
    print(f"[INFO] num_samples={num_samples}")
    print(f"[INFO] uff_inner_steps={uff_inner_steps}")
    print(f"[INFO] vina_inner_steps={vina_inner_steps}")
    print(f"[INFO] compute_uff_energy={compute_uff_energy}")
    print(f"[INFO] sanitize_output={sanitize_output}")
    print(f"[INFO] guidance_mol_copy={args.guidance_mol_copy}")
    print(f"[INFO] rdkit_io_mode={args.rdkit_io_mode}")
    print(f"[INFO] deterministic={bool(args.deterministic)}")
    print(f"[INFO] use_harmonic_prior={use_harmonic_prior}")
    print(f"[INFO] harmonic_spring_k={harmonic_spring_k:.6f}")
    print(f"[INFO] harmonic_langevin_steps={harmonic_langevin_steps}")
    print(f"[INFO] output_run_tag={run_tag}")

    # ===== template 후보 code 리스트 =====
    dude_list = [
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
    ]
    dude_set = set(dude_list)

    # ===== device =====
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("[Warning] CUDA not available. Running on CPU (this may be very slow).")

    # ===== model load =====
    model = DiffAlign()
    param_path = resolve_checkpoint_path(Path(args.checkpoint))
    if not param_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {param_path}")

    state_dict = torch.load(str(param_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"[INFO] checkpoint={param_path}")

    error_list = []
    LOG_PATH = args.log_path

    folder_names = [
        'CASP3', 'MK10', 'AOFB', 'ADRB2', 'ITAL', 'GRIA2', 'LKHA4', 'FAK1', 'MP2K1', 'FA10',
        'BRAF', 'SRC', 'PTN1', 'ABL1', 'FNTA', 'MCR', 'GLCM', 'HDAC8', 'PGH1', 'ADRB1', 'VGFR2',
        'PARP1', 'PYRD', 'PRGR', 'FKB1A', 'HIVPR', 'ANDR', 'RXRA', 'ESR1', 'MMP13', 'CP2C9',
        'HXK4', 'IGF1R', 'XIAP', 'PNPH', 'GCR', 'THRB', 'PUR2', 'CDK2', 'TRYB1', 'HS90A', 'EGFR',
        'MK01', 'LCK', 'TGFR1', 'GRIK1', 'FPPS', 'AKT1', 'DYR', 'PA2GA', 'RENI', 'PLK1', 'JAK2',
        'MET', 'AMPC', 'NRAM', 'CAH2', 'PPARG', 'WEE1', 'HMDH', 'HIVRT', 'PPARD', 'CXCR4', 'SAHH',
        'PPARA', 'ADA', 'ROCK1', 'BACE1', 'PGH2', 'HIVINT', 'MK14', 'ACES', 'PYGM', 'KIF11', 'ESR2',
        'THB', 'DPP4', 'ADA17', 'ACE', 'HDAC2', 'KITH', 'FABP4', 'TYSY', 'PDE5A', 'CSF1R', 'UROK'
    ]
    if args.job_manifest:
        all_jobs = load_manifest_jobs(Path(args.job_manifest).expanduser())
    else:
        all_jobs = discover_jobs_from_disco(folder_names, dude_set)

    global_query_count = len(all_jobs)
    selected_jobs = []
    for job in all_jobs:
        current_job_index = int(job["global_query_index"])
        if args.job_stride > 0:
            if (current_job_index % args.job_stride) != args.job_offset:
                continue
        else:
            if (current_job_index % args.num_shards) != args.shard_index:
                continue
        selected_jobs.append(job)
        if args.max_jobs > 0 and len(selected_jobs) >= args.max_jobs:
            break

    shard_selected_jobs = len(selected_jobs)
    shard_done_jobs = 0

    template_cache = {}
    broken_templates = set()

    for job in selected_jobs:
        folder_name = job["folder_name"]
        struct_dir = job["struct_dir"]
        template_name = job["template_name"]
        query_code = job["query_code"]
        current_job_index = int(job["global_query_index"])

        output_sdf_path = os.path.join(
            struct_dir,
            f"{query_code}_{run_tag}_{args.seed}.sdf",
        )
        legacy_output_sdf_path = os.path.join(
            struct_dir,
            f"{query_code}_320_wo_scheduling_{run_tag}_{args.seed}.sdf",
        )
        if args.skip_existing:
            skip_candidates = [output_sdf_path]
            if legacy_output_sdf_path != output_sdf_path:
                skip_candidates.append(legacy_output_sdf_path)
            skipped = False
            for existing_path in skip_candidates:
                if not os.path.exists(existing_path):
                    continue
                try:
                    if os.path.getsize(existing_path) > 0:
                        print(
                            f"[SKIP] shard_job={shard_done_jobs + 1} "
                            f"{folder_name}/{query_code} existing={existing_path}"
                        )
                        shard_done_jobs += 1
                        skipped = True
                        break
                except Exception:
                    continue
            if skipped:
                continue

        template_key = (job["template_sdf"], job["pocket_pdb"])
        if template_key in broken_templates:
            error_list.append((folder_name, template_name))
            continue

        if template_key not in template_cache:
            template_mol = load_first_mol_from_sdf(job["template_sdf"], sanitize=True, removeHs=True)
            if template_mol is None:
                broken_templates.add(template_key)
                error_list.append((folder_name, template_name))
                continue
            template_mol = force_remove_all_hydrogens(template_mol)
            if template_mol.GetNumAtoms() == 0:
                broken_templates.add(template_key)
                error_list.append((folder_name, template_name))
                continue

            try:
                pocket_mol = Chem.MolFromPDBFile(job["pocket_pdb"])
                if pocket_mol is None or pocket_mol.GetNumConformers() == 0:
                    raise ValueError("Failed to load pocket mol or missing conformer")
                pocket_mol = force_remove_all_hydrogens(pocket_mol)
                if pocket_mol.GetNumAtoms() == 0 or pocket_mol.GetNumConformers() == 0:
                    raise ValueError("Pocket mol became empty/invalid after force H removal")
            except Exception:
                broken_templates.add(template_key)
                error_list.append((folder_name, template_name))
                continue

            template_data = mol_to_graph_data_obj(template_mol)
            if template_data is None:
                broken_templates.add(template_key)
                error_list.append((folder_name, template_name))
                continue

            mean_pos = template_data.pos.mean(dim=0)
            template_data.pos = template_data.pos - mean_pos

            conf = pocket_mol.GetConformer(0)
            point_3d = Point3D(mean_pos[0].item(), mean_pos[1].item(), mean_pos[2].item())
            for i in range(pocket_mol.GetNumAtoms()):
                original_pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, original_pos - point_3d)

            template_batch = Batch.from_data_list([template_data.clone() for _ in range(num_samples)]).to(device)
            template_cache[template_key] = {
                "template_batch": template_batch,
                "pocket_mol": pocket_mol,
                "mean_pos": mean_pos,
            }

        template_ctx = template_cache[template_key]
        pocket_mol = template_ctx["pocket_mol"]
        mean_pos = template_ctx["mean_pos"]
        template_batch = template_ctx["template_batch"]

        print(folder_name, query_code, f"(template={template_name})")

        query_sdf_path = job["query_sdf"]
        query_mol = load_first_mol_from_sdf(query_sdf_path, sanitize=True, removeHs=True)
        if query_mol is None:
            error_list.append((folder_name, query_code))
            continue
        query_mol = force_remove_all_hydrogens(query_mol)
        if query_mol.GetNumAtoms() == 0:
            error_list.append((folder_name, query_code))
            continue

        query_data = mol_to_graph_data_obj(query_mol)
        if query_data is None:
            error_list.append((folder_name, query_code))
            continue

        query_data.pos = query_data.pos - mean_pos
        query_batch_list = [query_data.clone() for _ in range(num_samples)]
        query_batch = Batch.from_data_list(query_batch_list).to(device)

        ligand_pos_list = None
        calc_energy_list = None
        generated_mols = None
        pos_gen = None
        last_exc = None

        query_seed_base = base_seed + current_job_index * 1000
        force_disable_vina = (not vina_enabled)

        query_mols_use = None
        pocket_mols_use = None
        if args.guidance_mol_copy == "shallow":
            try:
                query_mol_guidance = Chem.Mol(query_mol)
                pocket_mol_guidance = Chem.Mol(pocket_mol)
            except Exception as e:
                last_exc = e
                log_sampling_failure(LOG_PATH, folder_name, query_code, -1, query_seed_base, e)
                error_list.append((folder_name, query_code))
                continue
            query_mols_use = [query_mol_guidance] * num_samples
            pocket_mols_use = [pocket_mol_guidance] * num_samples

        for attempt in range(5):
            attempt_seed = query_seed_base + attempt
            torch.manual_seed(attempt_seed)
            torch.cuda.manual_seed_all(attempt_seed)

            if args.guidance_mol_copy == "deep":
                try:
                    query_mols_use = [Chem.Mol(query_mol) for _ in range(num_samples)]
                    pocket_mols_use = [Chem.Mol(pocket_mol) for _ in range(num_samples)]
                except Exception as e:
                    last_exc = e
                    log_sampling_failure(LOG_PATH, folder_name, query_code, attempt, attempt_seed, e)
                    pos_gen = None
                    continue

            vina_scale = 0.0 if force_disable_vina else vina_guidance_scale

            try:
                pos_gen, _ = model.DDPM_Sampling_Heun_UFF_vina_wo_scheduling(
                    query_batch=query_batch,
                    reference_batch=template_batch,
                    use_self_condition=use_self_condition,
                    self_condition_use_steered_x0=self_condition_use_steered_x0,
                    query_mols=query_mols_use,
                    pocket_mols=pocket_mols_use,
                    noise_temperature=0.5,
                    use_harmonic_prior=use_harmonic_prior,
                    harmonic_spring_k=harmonic_spring_k,
                    harmonic_langevin_steps=harmonic_langevin_steps,
                    uff_guidance_scale=uff_guidance_scale,
                    uff_inner_steps=uff_inner_steps,
                    vina_guidance_scale=vina_scale,
                    vina_inner_steps=vina_inner_steps
                )

                if is_bad_pos_tensor(pos_gen):
                    raise RuntimeError("pos_gen is None/empty/NaN/inf (sampling failed)")

            except torch.cuda.OutOfMemoryError as e:
                last_exc = e
                log_sampling_failure(LOG_PATH, folder_name, query_code, attempt, attempt_seed, e)
                pos_gen = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            except Exception as e:
                last_exc = e
                log_sampling_failure(LOG_PATH, folder_name, query_code, attempt, attempt_seed, e)

                tb = traceback.format_exc()
                if ("vinasf_torch" in tb) or ("score_and_gradient" in tb) or ("vina_model" in tb):
                    force_disable_vina = True

                pos_gen = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            generated_poses_cpu = pos_gen.detach().cpu() + mean_pos
            ligand_pos_list = []
            ptr = query_batch.ptr.cpu().tolist()
            for i in range(num_samples):
                ligand_pos_list.append(generated_poses_cpu[ptr[i]:ptr[i + 1]])

            if args.rdkit_io_mode == "optimized":
                generated_mols = []
                for i in range(num_samples):
                    current_pose = ligand_pos_list[i]
                    try:
                        gen_mol_rdkit = set_rdmol_positions(query_mol, current_pose)
                    except Exception:
                        gen_mol_rdkit = None
                    generated_mols.append(gen_mol_rdkit)

                if compute_uff_energy:
                    calc_energy_list = []
                    for gen_mol_rdkit in generated_mols:
                        if gen_mol_rdkit is None:
                            calc_energy_list.append(float("inf"))
                            continue

                        try:
                            ff = AllChem.UFFGetMoleculeForceField(gen_mol_rdkit)
                            calc_energy_list.append(float(ff.CalcEnergy()))
                        except Exception:
                            calc_energy_list.append(float("inf"))

                    if (len(calc_energy_list) == 0) or (min(calc_energy_list) >= 1e11):
                        pos_gen = None
                        ligand_pos_list = None
                        calc_energy_list = None
                        continue
                else:
                    calc_energy_list = [float("nan")] * num_samples
            else:
                generated_mols = None
                if compute_uff_energy:
                    calc_energy_list = []
                    for i in range(num_samples):
                        current_pose = ligand_pos_list[i]
                        try:
                            gen_mol_rdkit = set_rdmol_positions(query_mol, current_pose)
                        except Exception:
                            gen_mol_rdkit = None

                        if gen_mol_rdkit is None:
                            calc_energy_list.append(float("inf"))
                            continue

                        try:
                            ff = AllChem.UFFGetMoleculeForceField(gen_mol_rdkit)
                            calc_energy_list.append(float(ff.CalcEnergy()))
                        except Exception:
                            calc_energy_list.append(float("inf"))

                    if (len(calc_energy_list) == 0) or (min(calc_energy_list) >= 1e11):
                        pos_gen = None
                        ligand_pos_list = None
                        calc_energy_list = None
                        continue
                else:
                    calc_energy_list = [float("nan")] * num_samples

            break

        if (pos_gen is None) or (ligand_pos_list is None) or (calc_energy_list is None):
            print(f"[ERROR] {folder_name}/{query_code}: sampling failed after retries. last_exc={repr(last_exc)}")
            error_list.append((folder_name, query_code))
            continue

        print(f"Saving all {num_samples} generated poses to {output_sdf_path}")

        with Chem.SDWriter(output_sdf_path) as w:
            if args.rdkit_io_mode == "optimized":
                iterable = enumerate(generated_mols if generated_mols is not None else [])
            else:
                iterable = enumerate(ligand_pos_list)

            for idx, item in iterable:
                if args.rdkit_io_mode == "optimized":
                    gen_mol = item
                else:
                    current_pose = item
                    try:
                        gen_mol = set_rdmol_positions(query_mol, current_pose)
                    except Exception:
                        gen_mol = None

                if gen_mol is None:
                    print(f"[Warning] Failed to create RDKit mol for pose {idx}. Skipping.")
                    continue

                gen_mol.SetProp("Pose_ID", str(idx + 1))
                if compute_uff_energy:
                    uff_energy = calc_energy_list[idx]
                    gen_mol.SetProp("UFF_Energy", f"{uff_energy:.4f}")
                gen_mol.SetProp("UFF_Guidance_Scale", f"{uff_guidance_scale:.6f}")
                gen_mol.SetProp("Vina_Guidance_Scale", f"{vina_guidance_scale:.6f}")
                gen_mol.SetProp("Vina_Mode", vina_tag)
                gen_mol.SetProp("Use_Self_Condition", str(int(use_self_condition)))
                gen_mol.SetProp(
                    "Self_Condition_Use_Steered_X0",
                    str(int(self_condition_use_steered_x0)),
                )
                gen_mol.SetProp("Self_Condition_Source", self_condition_source)
                gen_mol.SetProp("Use_Harmonic_Prior", str(int(use_harmonic_prior)))
                gen_mol.SetProp("Harmonic_Spring_K", f"{harmonic_spring_k:.6f}")
                gen_mol.SetProp("Harmonic_Langevin_Steps", str(harmonic_langevin_steps))
                gen_mol.SetProp("Checkpoint", str(param_path))
                gen_mol.SetProp("Run_Tag", run_tag)

                if sanitize_output:
                    try:
                        Chem.SanitizeMol(gen_mol)
                    except Exception as sanitize_e:
                        print(f"Error during sanitization for pose {idx}: {sanitize_e}. Skipping this pose.")
                        continue

                w.write(gen_mol)
        shard_done_jobs += 1

    print("===== DONE =====")
    print(f"[INFO] global_query_count={global_query_count}")
    print(f"[INFO] shard_selected_jobs={shard_selected_jobs}")
    print(f"[INFO] shard_done_jobs={shard_done_jobs}")
    print("Errors:", error_list)


if __name__ == "__main__":
    main()
