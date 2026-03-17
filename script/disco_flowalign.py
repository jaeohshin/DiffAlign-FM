#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disco_flowalign.py
------------------
FlowAlign benchmark on the DISCO dataset.
Adapted from the DiffAlign author's benchmark script.
Key changes:
  - Loads FlowAlign (flow matching) instead of DiffAlign (diffusion)
  - Calls FM_Sampling instead of DDPM_Sampling_Heun_UFF_vina_wo_scheduling
  - Outputs go to ./output_disco/{folder_name}/
"""

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

import sys
sys.path.insert(0, "/store/jaeohshin/work/Diffalign")
from models.epsnet.flow import FlowAlign
from utils.chem import *
import sys
sys.path.insert(0, '/store/jaeohshin/work/Diffalign')


# ===== RDKit 로그 끄기 =====
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def load_first_mol_from_sdf(sdf_path: str, *, sanitize: bool = True, removeHs: bool = True):
    if not os.path.exists(sdf_path):
        return None
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=removeHs)
    for m in suppl:
        if m is not None:
            return m
    return None


def force_remove_all_hydrogens(mol):
    if mol is None:
        return None
    rw_mol = Chem.RWMol(Chem.Mol(mol))
    hydrogen_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 1]
    if not hydrogen_indices:
        return Chem.Mol(mol)
    for idx in reversed(hydrogen_indices):
        rw_mol.RemoveAtom(idx)
    stripped = rw_mol.GetMol()
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
        atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.long)

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
    if x is None:
        return True
    if (not torch.is_tensor(x)) or x.numel() == 0:
        return True
    if not torch.isfinite(x).all():
        return True
    return False


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
        struct_dir = f"./data/disco/{folder_name}/PDB_Structures"
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
            jobs.append({
                "global_query_index": global_idx,
                "folder_name": folder_name,
                "struct_dir": struct_dir,
                "template_name": template_name,
                "query_code": query_code,
                "template_sdf": template_sdf_path,
                "pocket_pdb": pocket_path,
                "query_sdf": os.path.join(struct_dir, file_name),
            })
            global_idx += 1
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/100.pt",
        help="FlowAlign checkpoint path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uff-guidance-scale", type=float, default=0.05)
    parser.add_argument("--uff-inner-steps", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of Euler ODE steps for FM_Sampling.")
    parser.add_argument("--noise-temperature", type=float, default=0.0,
                        help="0.0 = deterministic ODE (recommended for benchmarking).")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--job-manifest", type=str, default="")
    parser.add_argument("--job-stride", type=int, default=0)
    parser.add_argument("--job-offset", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-path", type=str, default="./flowalign_sampling_failures.log")
    parser.add_argument("--output-tag", type=str, default="flowalign")
    parser.add_argument(
        "--compute-uff-energy", type=int, default=1, choices=[0, 1],
    )
    parser.add_argument(
        "--sanitize-output", type=int, default=1, choices=[0, 1],
    )
    args = parser.parse_args()

    # ===== validation =====
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.job_stride > 0:
        if args.job_offset < 0 or args.job_offset >= args.job_stride:
            raise ValueError("--job-offset must satisfy 0 <= job-offset < job-stride when job-stride > 0")

    base_seed = int(args.seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    uff_guidance_scale = float(args.uff_guidance_scale)
    num_samples = int(args.num_samples)
    uff_inner_steps = int(args.uff_inner_steps)
    num_steps = int(args.num_steps)
    noise_temperature = float(args.noise_temperature)
    compute_uff_energy = bool(args.compute_uff_energy)
    sanitize_output = bool(args.sanitize_output)
    run_tag = args.output_tag.strip().replace(" ", "_")

    print(f"[INFO] checkpoint={args.checkpoint}")
    print(f"[INFO] uff_guidance_scale={uff_guidance_scale}")
    print(f"[INFO] uff_inner_steps={uff_inner_steps}")
    print(f"[INFO] num_steps={num_steps}")
    print(f"[INFO] noise_temperature={noise_temperature}")
    print(f"[INFO] num_samples={num_samples}")
    print(f"[INFO] output_tag={run_tag}")

    # ===== template candidates =====
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Warning] CUDA not available. Running on CPU.")

    # ===== model load =====
    model = FlowAlign()
    param_path = resolve_checkpoint_path(Path(args.checkpoint))
    state_dict = torch.load(str(param_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded FlowAlign checkpoint: {param_path}")

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
    error_list = []
    LOG_PATH = args.log_path

    template_cache = {}
    broken_templates = set()

    for job in selected_jobs:
        folder_name = job["folder_name"]
        struct_dir = job["struct_dir"]
        template_name = job["template_name"]
        query_code = job["query_code"]
        current_job_index = int(job["global_query_index"])

        # output goes to ./output_disco/{folder_name}/
        out_dir = os.path.join("./output_disco", folder_name)
        os.makedirs(out_dir, exist_ok=True)
        output_sdf_path = os.path.join(out_dir, f"{query_code}_{run_tag}_{args.seed}.sdf")

        if args.skip_existing and os.path.exists(output_sdf_path):
            try:
                if os.path.getsize(output_sdf_path) > 0:
                    print(f"[SKIP] {folder_name}/{query_code} existing={output_sdf_path}")
                    shard_done_jobs += 1
                    continue
            except Exception:
                pass

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

        print(f"{folder_name} {query_code} (template={template_name})")

        query_mol = load_first_mol_from_sdf(job["query_sdf"], sanitize=True, removeHs=True)
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
        query_batch = Batch.from_data_list([query_data.clone() for _ in range(num_samples)]).to(device)

        pos_gen = None
        ligand_pos_list = None
        calc_energy_list = None
        last_exc = None

        query_seed_base = base_seed + current_job_index * 1000

        for attempt in range(5):
            attempt_seed = query_seed_base + attempt
            torch.manual_seed(attempt_seed)
            torch.cuda.manual_seed_all(attempt_seed)

            query_mols_use = [Chem.Mol(query_mol) for _ in range(num_samples)]
            pocket_mols_use = [Chem.Mol(pocket_mol) for _ in range(num_samples)]

            try:
                pos_gen, _ = model.FM_Sampling(
                    query_batch=query_batch,
                    reference_batch=template_batch,
                    num_steps=num_steps,
                    query_mols=query_mols_use,
                    pocket_mols=pocket_mols_use,
                    uff_guidance_scale=uff_guidance_scale,
                    uff_inner_steps=uff_inner_steps,
                    noise_temperature=noise_temperature,
                )

                if is_bad_pos_tensor(pos_gen):
                    raise RuntimeError("pos_gen is None/empty/NaN/inf")

            except torch.cuda.OutOfMemoryError as e:
                last_exc = e
                log_sampling_failure(LOG_PATH, folder_name, query_code, attempt, attempt_seed, e)
                pos_gen = None
                torch.cuda.empty_cache()
                continue

            except Exception as e:
                last_exc = e
                log_sampling_failure(LOG_PATH, folder_name, query_code, attempt, attempt_seed, e)
                pos_gen = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # split per-sample positions
            generated_poses_cpu = pos_gen.detach().cpu() + mean_pos
            ligand_pos_list = []
            ptr = query_batch.ptr.cpu().tolist()
            for i in range(num_samples):
                ligand_pos_list.append(generated_poses_cpu[ptr[i]:ptr[i + 1]])

            # build mols and compute UFF energy
            generated_mols = []
            for i in range(num_samples):
                try:
                    gen_mol_rdkit = set_rdmol_positions(query_mol, ligand_pos_list[i])
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

            break  # success

        if pos_gen is None or ligand_pos_list is None or calc_energy_list is None:
            print(f"[ERROR] {folder_name}/{query_code}: sampling failed. last_exc={repr(last_exc)}")
            error_list.append((folder_name, query_code))
            continue

        print(f"Saving {num_samples} poses to {output_sdf_path}")
        with Chem.SDWriter(output_sdf_path) as w:
            for idx, gen_mol in enumerate(generated_mols):
                if gen_mol is None:
                    print(f"[Warning] Pose {idx} is None, skipping.")
                    continue

                gen_mol.SetProp("Pose_ID", str(idx + 1))
                if compute_uff_energy:
                    gen_mol.SetProp("UFF_Energy", f"{calc_energy_list[idx]:.4f}")
                gen_mol.SetProp("UFF_Guidance_Scale", f"{uff_guidance_scale:.6f}")
                gen_mol.SetProp("Num_Steps", str(num_steps))
                gen_mol.SetProp("Noise_Temperature", f"{noise_temperature:.6f}")
                gen_mol.SetProp("Checkpoint", str(param_path))
                gen_mol.SetProp("Run_Tag", run_tag)

                if sanitize_output:
                    try:
                        Chem.SanitizeMol(gen_mol)
                    except Exception as sanitize_e:
                        print(f"[Warning] Sanitization failed for pose {idx}: {sanitize_e}. Skipping.")
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