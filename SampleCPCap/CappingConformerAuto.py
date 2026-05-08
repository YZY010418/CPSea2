#!/usr/bin/env python3
"""Auto-determine conformer count by iterative generate/relax/filter/cluster."""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina


_RMSD_WORKER_MOLS = None
_RMSD_WORKER_ATOM_MAP = None
SDF_SUFFIXES = {".sdf", ".sd"}
BATCH_HISTORY_FIELDS = [
    "batch",
    "seed",
    "generated_count",
    "relaxed_count",
    "energy_evaluable_total",
    "energy_keep_percent",
    "energy_cutoff",
    "energy_kept_count",
    "accepted_total",
    "cluster_count",
    "cluster_gain",
    "low_gain_streak",
]
CONFORMER_SUMMARY_FIELDS = [
    "entry_id",
    "batch",
    "source_file",
    "archive_file",
    "accepted_file",
    "energy_kj_per_mol",
]


@dataclass(frozen=True)
class WorkflowPaths:
    root: Path
    archive: Path
    accepted: Path
    batches: Path


def run_cmd(cmd, cwd=None):
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        command_text = " ".join(str(token) for token in cmd)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command_text}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def parse_energy_kj(mol):
    for key in (
        "Energy_kJ_per_mol",
        "r_mmod_Potential_Energy-OPLS4",
        "r_mmod_Potential_Energy",
        "r_ff_Potential_Energy-OPLS4",
    ):
        if not mol.HasProp(key):
            continue
        value = mol.GetProp(key).strip()
        try:
            return float(value)
        except ValueError:
            continue
    return None


def fallback_mmff_energy_kj(mol):
    try:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        if ff is None:
            return None
        return float(ff.CalcEnergy())
    except Exception:
        return None


def read_single_sdf(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def validate_same_topology(mols):
    if not mols:
        return
    template = tuple(atom.GetAtomicNum() for atom in mols[0].GetAtoms())
    for idx, mol in enumerate(mols[1:], start=1):
        if tuple(atom.GetAtomicNum() for atom in mol.GetAtoms()) != template:
            raise ValueError(
                f"Molecule topology mismatch before clustering at index {idx}: "
                "atom ordering/types differ."
            )


def fixed_heavy_atom_map(mol):
    atom_map = [
        (atom.GetIdx(), atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not atom_map:
        raise ValueError("No heavy atoms available for RMSD clustering")
    return atom_map


def fixed_order_heavy_rmsd(mol_i, mol_j, atom_map):
    rmsd, _ = rdMolAlign.GetAlignmentTransform(mol_i, mol_j, atomMap=atom_map)
    return float(rmsd)


def init_rmsd_worker(mols, atom_map):
    global _RMSD_WORKER_MOLS
    global _RMSD_WORKER_ATOM_MAP
    _RMSD_WORKER_MOLS = mols
    _RMSD_WORKER_ATOM_MAP = atom_map


def rmsd_distance_chunk(index_pairs):
    return [
        (i, j, fixed_order_heavy_rmsd(_RMSD_WORKER_MOLS[i], _RMSD_WORKER_MOLS[j], _RMSD_WORKER_ATOM_MAP))
        for i, j in index_pairs
    ]


def serial_pairwise_rmsd_results(mols, index_pairs, atom_map):
    return [
        (i, j, fixed_order_heavy_rmsd(mols[i], mols[j], atom_map))
        for i, j in index_pairs
    ]


def compute_pairwise_rmsd_results(mols, index_pairs, atom_map, processes=1):
    if not index_pairs:
        return []

    worker_count = min(max(1, int(processes)), len(index_pairs))
    if worker_count == 1:
        return serial_pairwise_rmsd_results(mols, index_pairs, atom_map)

    pair_count = len(index_pairs)
    chunk_size = max(100, min(5000, pair_count // (worker_count * 8) or 1))
    chunks = [index_pairs[i:i + chunk_size] for i in range(0, len(index_pairs), chunk_size)]

    results = []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_rmsd_worker,
            initargs=(mols, atom_map),
        ) as executor:
            for chunk_results in executor.map(rmsd_distance_chunk, chunks):
                results.extend(chunk_results)
    except Exception as exc:
        print(
            f"Warning: parallel RMSD clustering failed ({exc}); falling back to serial.",
            file=sys.stderr,
        )
        results = serial_pairwise_rmsd_results(mols, index_pairs, atom_map)
    return results


def rmsd_cache_key(entry_a, entry_b):
    id_a = entry_a["entry_id"]
    id_b = entry_b["entry_id"]
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


def cached_pairwise_rmsd_distances(entries, rmsd_cache, processes=1):
    mols = [entry["mol"] for entry in entries]
    n_mols = len(mols)
    if n_mols <= 1:
        return []

    validate_same_topology(mols)
    atom_map = fixed_heavy_atom_map(mols[0])

    missing_pairs = [
        (i, j)
        for i in range(1, n_mols)
        for j in range(i)
        if rmsd_cache_key(entries[i], entries[j]) not in rmsd_cache
    ]

    for i, j, distance in compute_pairwise_rmsd_results(mols, missing_pairs, atom_map, processes=processes):
        key = rmsd_cache_key(entries[i], entries[j])
        rmsd_cache[key] = distance

    return [
        rmsd_cache[rmsd_cache_key(entries[i], entries[j])]
        for i in range(1, n_mols)
        for j in range(i)
    ]


def cluster_count_from_distances(distances, n_mols, rmsd_cutoff):
    if n_mols == 0:
        return 0
    if n_mols == 1:
        return 1

    expected_distances = n_mols * (n_mols - 1) // 2
    if len(distances) != expected_distances:
        raise ValueError(
            f"Expected {expected_distances} pairwise distances for {n_mols} molecules, "
            f"got {len(distances)}."
        )

    clusters = Butina.ClusterData(
        distances,
        nPts=n_mols,
        distThresh=float(rmsd_cutoff),
        isDistData=True,
        reordering=True,
    )
    return len(clusters)


def collect_relaxed_sdfs(relaxed_dir):
    return [
        path
        for path in sorted(Path(relaxed_dir).iterdir())
        if path.is_file() and path.suffix.lower() in SDF_SUFFIXES
    ]


def count_sdfs(path):
    return len(collect_relaxed_sdfs(path))


def copy_cap_sidecars(source_dir, target_dir):
    for path in Path(source_dir).glob("*_atom_names.csv"):
        target = Path(target_dir) / path.name
        if not target.exists():
            shutil.copy2(path, target)


def archive_relaxed_sdfs(relaxed_dir, archive_dir, batch_idx, start_entry_id):
    archived_entries = []
    for path in collect_relaxed_sdfs(relaxed_dir):
        mol = read_single_sdf(path)
        if mol is None:
            continue
        energy = parse_energy_kj(mol)
        if energy is None:
            energy = fallback_mmff_energy_kj(mol)
        if energy is None:
            continue

        archive_path = Path(archive_dir) / path.name
        shutil.copy2(path, archive_path)

        archived_entries.append(
            {
                "entry_id": start_entry_id + len(archived_entries),
                "batch": batch_idx,
                "source_file": str(path),
                "archive_file": str(archive_path),
                "accepted_file": "",
                "energy_kj_per_mol": energy,
                "mol": mol,
            }
        )
    copy_cap_sidecars(relaxed_dir, archive_dir)
    return archived_entries


def clear_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def select_lowest_energy_percent(entries, keep_percent):
    if not entries:
        return [], None
    keep_fraction = float(keep_percent) / 100.0
    keep_count = int(math.ceil(len(entries) * keep_fraction))
    keep_count = max(1, min(len(entries), keep_count))
    sorted_entries = sorted(entries, key=lambda entry: entry["energy_kj_per_mol"])
    selected = sorted_entries[:keep_count]
    return selected, selected[-1]["energy_kj_per_mol"]


def rebuild_accepted_dir(selected_entries, all_entries, accepted_dir, archive_dir):
    clear_directory(accepted_dir)
    for entry in all_entries:
        entry["accepted_file"] = ""
    for entry in selected_entries:
        source = Path(entry["archive_file"])
        target = Path(accepted_dir) / source.name
        shutil.copy2(source, target)
        entry["accepted_file"] = str(target)
    copy_cap_sidecars(archive_dir, accepted_dir)


def write_batch_history(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_conformer_summary(path, entries):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFORMER_SUMMARY_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "entry_id": entry["entry_id"],
                    "batch": entry["batch"],
                    "source_file": entry["source_file"],
                    "archive_file": entry["archive_file"],
                    "accepted_file": entry["accepted_file"],
                    "energy_kj_per_mol": f"{entry['energy_kj_per_mol']:.6f}",
                }
            )


def build_schrodinger_run_path(explicit_path=None):
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    env_value = os.environ.get("SCHRODINGER")
    if env_value:
        return (Path(env_value).expanduser().resolve() / "run")
    return Path("run")


def make_workflow_paths(output_root):
    root = Path(output_root).expanduser().resolve()
    paths = WorkflowPaths(
        root=root,
        archive=root / "all_relaxed_sdf",
        accepted=root / "accepted_energy_filtered_sdf",
        batches=root / "batches",
    )
    for path in (paths.root, paths.archive, paths.accepted, paths.batches):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_batch_dirs(paths, batch_idx):
    batch_dir = paths.batches / f"batch_{batch_idx:04d}"
    generated_dir = batch_dir / "generated"
    relaxed_dir = batch_dir / "relaxed"
    generated_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, generated_dir, relaxed_dir


def run_capping_generation(args, capping_gen_script, batch_name, generated_dir, seed):
    run_cmd([
        args.python, str(capping_gen_script),
        "--name", batch_name,
        "--smiles", args.smiles,
        "--num", str(args.batch_size),
        "--out", str(generated_dir),
        "--seed", str(seed),
    ])


def run_relaxation(args, schrodinger_run, relax_light_script, generated_dir, relaxed_dir):
    run_cmd([
        str(schrodinger_run), "python3", str(relax_light_script),
        "-i", str(generated_dir),
        "-o", str(relaxed_dir),
        "--ligand-asl", args.ligand_asl,
        "--hydrogen-scope", args.hydrogen_scope,
        "--max-steps", str(args.max_steps),
        "--grad-conv", str(args.grad_conv),
        "--dielectric", str(args.dielectric),
        "-p", str(args.processors),
    ])


def build_batch_row(
    batch_idx,
    seed,
    generated_dir,
    relaxed_dir,
    all_relaxed_entries,
    accepted_entries,
    energy_keep_percent,
    energy_cutoff,
    cluster_count,
    cluster_gain,
    low_gain_streak,
):
    return {
        "batch": batch_idx,
        "seed": seed,
        "generated_count": count_sdfs(generated_dir),
        "relaxed_count": count_sdfs(relaxed_dir),
        "energy_evaluable_total": len(all_relaxed_entries),
        "energy_keep_percent": f"{energy_keep_percent:.6g}",
        "energy_cutoff": "" if energy_cutoff is None else f"{energy_cutoff:.6f}",
        "energy_kept_count": len(accepted_entries),
        "accepted_total": len(accepted_entries),
        "cluster_count": cluster_count,
        "cluster_gain": cluster_gain,
        "low_gain_streak": low_gain_streak,
    }


def print_batch_status(row, energy_cutoff):
    energy_cutoff_text = "NA" if energy_cutoff is None else f"{energy_cutoff:.3f}"
    print(
        f"Generated: {row['generated_count']}, relaxed: {row['relaxed_count']}, "
        f"energy-evaluable(total): {row['energy_evaluable_total']}, "
        f"energy-kept: {row['energy_kept_count']}, "
        f"energy cutoff: {energy_cutoff_text}, "
        f"clusters(total): {row['cluster_count']}, "
        f"cluster gain: {row['cluster_gain']}, low-gain streak: {row['low_gain_streak']}"
    )


def write_final_outputs(paths, all_relaxed_entries, accepted_entries, batch_rows):
    batch_history_csv = paths.root / "batch_history.csv"
    all_relaxed_csv = paths.root / "all_relaxed_conformers.csv"
    accepted_csv = paths.root / "accepted_energy_filtered_conformers.csv"

    write_batch_history(batch_history_csv, batch_rows)
    write_conformer_summary(all_relaxed_csv, all_relaxed_entries)
    write_conformer_summary(accepted_csv, accepted_entries)
    return batch_history_csv, all_relaxed_csv, accepted_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iteratively generate, relax, energy-filter, and cluster cap conformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("required inputs")
    for flag, help_text in (
        ("--name", "Cap name prefix."),
        ("--smiles", "Cap SMILES."),
        ("--out", "Output root directory."),
    ):
        inputs.add_argument(flag, required=True, help=help_text)

    sampling = parser.add_argument_group("sampling and stopping")
    sampling.add_argument("--batch-size", type=int, default=1000, help="Conformers generated per batch.")
    sampling.add_argument("--max-batches", type=int, default=50, help="Maximum number of batches.")
    sampling.add_argument("--seed-start", type=int, default=42, help="First batch seed; each next batch uses +1.")
    sampling.add_argument("--cluster-rmsd", type=float, default=0.5, help="RMSD threshold for Butina clustering.")
    sampling.add_argument("--min-cluster-increase", type=int, default=5, help="Low-gain threshold.")
    sampling.add_argument("--patience", type=int, default=3, help="Consecutive low-gain batches before stopping.")
    sampling.add_argument("--energy-keep-percent", type=float, default=90.0, help="Lowest-energy percent to keep; 100 disables filtering.")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--schrodinger-run", default=None, help="Schrodinger run launcher; defaults to $SCHRODINGER/run or run on PATH.")
    runtime.add_argument("--python", default=sys.executable, help="Python executable for CappingGen.py.")
    runtime.add_argument("-p", "--processors", type=int, default=1, help="Processors for relax_light.py.")
    runtime.add_argument("--cluster-processors", type=int, default=None, help="Processes for RMSD clustering; defaults to --processors.")

    relaxation = parser.add_argument_group("relax_light controls")
    relaxation.add_argument("--ligand-asl", default='chain.name "A"', help="Ligand ASL for relax_light.py.")
    relaxation.add_argument("--hydrogen-scope", choices=["all", "ligand"], default="all")
    relaxation.add_argument("--max-steps", type=int, default=200)
    relaxation.add_argument("--grad-conv", type=float, default=0.05)
    relaxation.add_argument("--dielectric", type=float, default=80.0)

    parser.add_argument("--keep-batches", action="store_true", help="Keep generated/relaxed files for each batch.")
    args = parser.parse_args()
    for attr, label in (
        ("batch_size", "--batch-size"),
        ("max_batches", "--max-batches"),
        ("patience", "--patience"),
        ("processors", "--processors"),
    ):
        if getattr(args, attr) < 1:
            parser.error(f"{label} must be >= 1")
    if args.cluster_rmsd <= 0:
        parser.error("--cluster-rmsd must be > 0")
    if args.min_cluster_increase < 0:
        parser.error("--min-cluster-increase must be >= 0")
    if args.cluster_processors is not None and args.cluster_processors < 1:
        parser.error("--cluster-processors must be >= 1")
    if not (0.0 < args.energy_keep_percent <= 100.0):
        parser.error("--energy-keep-percent must be > 0 and <= 100")
    return args


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    capping_gen_script = script_dir / "CappingGen.py"
    relax_light_script = script_dir / "relax_light.py"
    schrodinger_run = build_schrodinger_run_path(args.schrodinger_run)
    cluster_processes = args.cluster_processors or args.processors
    paths = make_workflow_paths(args.out)

    all_relaxed_entries = []
    accepted_entries = []
    batch_rows = []
    low_gain_streak = 0
    previous_cluster_count = 0
    rmsd_cache = {}

    for batch_idx in range(1, args.max_batches + 1):
        seed = args.seed_start + batch_idx - 1
        batch_name = f"{args.name}_b{batch_idx:04d}"
        batch_dir, generated_dir, relaxed_dir = make_batch_dirs(paths, batch_idx)

        print(f"\n=== Batch {batch_idx} ===")
        print(f"Seed: {seed}")
        print("Step 1/4: generating conformers")
        run_capping_generation(args, capping_gen_script, batch_name, generated_dir, seed)

        print("Step 2/4: relaxing conformers")
        run_relaxation(args, schrodinger_run, relax_light_script, generated_dir, relaxed_dir)

        print(f"Step 3/4: archiving relaxed conformers and keeping lowest {args.energy_keep_percent:g}% by energy")
        batch_archived = archive_relaxed_sdfs(
            relaxed_dir,
            paths.archive,
            batch_idx,
            start_entry_id=len(all_relaxed_entries),
        )
        all_relaxed_entries.extend(batch_archived)
        accepted_entries, energy_cutoff = select_lowest_energy_percent(
            all_relaxed_entries,
            args.energy_keep_percent,
        )
        rebuild_accepted_dir(accepted_entries, all_relaxed_entries, paths.accepted, paths.archive)

        print(f"Step 4/4: clustering accepted conformers ({cluster_processes} process(es))")
        cluster_count = cluster_count_from_distances(
            cached_pairwise_rmsd_distances(accepted_entries, rmsd_cache, cluster_processes),
            len(accepted_entries),
            args.cluster_rmsd,
        )
        cluster_gain = cluster_count - previous_cluster_count
        previous_cluster_count = cluster_count

        if cluster_gain < args.min_cluster_increase:
            low_gain_streak += 1
        else:
            low_gain_streak = 0

        row = build_batch_row(
            batch_idx,
            seed,
            generated_dir,
            relaxed_dir,
            all_relaxed_entries,
            accepted_entries,
            args.energy_keep_percent,
            energy_cutoff,
            cluster_count,
            cluster_gain,
            low_gain_streak,
        )
        batch_rows.append(row)
        print_batch_status(row, energy_cutoff)

        if not args.keep_batches:
            shutil.rmtree(batch_dir, ignore_errors=True)

        if low_gain_streak >= args.patience:
            print(
                f"Stopping early: cluster increase < {args.min_cluster_increase} "
                f"for {args.patience} consecutive batches."
            )
            break

    batch_history_csv, all_relaxed_csv, accepted_csv = write_final_outputs(
        paths,
        all_relaxed_entries,
        accepted_entries,
        batch_rows,
    )

    print("\n=== Auto Conformer Summary ===")
    print(f"Relaxed conformers with readable energy: {len(all_relaxed_entries)}")
    print(f"Accepted energy-filtered conformers: {len(accepted_entries)}")
    if batch_rows:
        print(f"Final cluster count: {batch_rows[-1]['cluster_count']}")
        print(f"Batches run: {batch_rows[-1]['batch']}")
    print(f"All relaxed SDF directory: {paths.archive}")
    print(f"Accepted SDF directory: {paths.accepted}")
    print(f"Batch history CSV: {batch_history_csv}")
    print(f"All relaxed conformer CSV: {all_relaxed_csv}")
    print(f"Accepted conformer CSV: {accepted_csv}")


if __name__ == "__main__":
    main()
