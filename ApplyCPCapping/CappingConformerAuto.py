#!/usr/bin/env python3
"""Auto-determine conformer count by iterative generate/relax/filter/cluster."""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina


_RMSD_WORKER_MOLS = None


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
        atom_types = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
        if atom_types != template:
            raise ValueError(
                f"Molecule topology mismatch before clustering at index {idx}: "
                "atom ordering/types differ."
            )


def init_rmsd_worker(mols):
    global _RMSD_WORKER_MOLS
    _RMSD_WORKER_MOLS = mols


def rmsd_distance_chunk(index_pairs):
    return [
        float(rdMolAlign.GetBestRMS(_RMSD_WORKER_MOLS[i], _RMSD_WORKER_MOLS[j]))
        for i, j in index_pairs
    ]


def iter_pair_index_chunks(n_mols, start_i, chunk_size):
    chunk = []
    for i in range(start_i, n_mols):
        for j in range(i):
            chunk.append((i, j))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def serial_pairwise_rmsd_distances(mols, start_i=1):
    distances = []
    for i in range(start_i, len(mols)):
        for j in range(i):
            rmsd = rdMolAlign.GetBestRMS(mols[i], mols[j])
            distances.append(float(rmsd))
    return distances


def pairwise_rmsd_distances(mols, start_i=1, processes=1):
    n_mols = len(mols)
    start_i = max(1, int(start_i))
    if n_mols <= 1 or start_i >= n_mols:
        return []

    worker_count = min(max(1, int(processes)), n_mols - 1)
    if worker_count == 1:
        return serial_pairwise_rmsd_distances(mols, start_i=start_i)

    pair_count = (start_i + n_mols - 1) * (n_mols - start_i) // 2
    chunk_size = max(100, min(5000, pair_count // (worker_count * 8) or 1))
    chunks = iter_pair_index_chunks(n_mols, start_i, chunk_size)

    distances = []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_rmsd_worker,
            initargs=(mols,),
        ) as executor:
            for chunk_distances in executor.map(rmsd_distance_chunk, chunks):
                distances.extend(chunk_distances)
    except Exception as exc:
        print(
            f"Warning: parallel RMSD clustering failed ({exc}); falling back to serial.",
            file=sys.stderr,
        )
        distances = serial_pairwise_rmsd_distances(mols, start_i=start_i)
    return distances


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


def pairwise_rmsd_cluster_count(mols, rmsd_cutoff, processes=1):
    validate_same_topology(mols)
    distances = pairwise_rmsd_distances(mols, processes=processes)
    return cluster_count_from_distances(distances, len(mols), rmsd_cutoff)


def collect_relaxed_sdfs(relaxed_dir):
    files = []
    for path in sorted(Path(relaxed_dir).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".sdf", ".sd"}:
            continue
        files.append(path)
    return files


def filter_negative_relaxed_sdfs(relaxed_dir, accepted_dir, batch_idx):
    accepted_entries = []
    copied_sidecars = set()
    for path in collect_relaxed_sdfs(relaxed_dir):
        mol = read_single_sdf(path)
        if mol is None:
            continue
        energy = parse_energy_kj(mol)
        if energy is None:
            energy = fallback_mmff_energy_kj(mol)
        if energy is None or energy >= 0.0:
            continue

        out_name = path.name
        out_path = Path(accepted_dir) / out_name
        shutil.copy2(path, out_path)

        stem = path.stem
        if "_conformers_" in stem:
            cap_prefix = stem.split("_conformers_", 1)[0]
            sidecar_name = f"{cap_prefix}_atom_names.csv"
            sidecar_src = Path(relaxed_dir) / sidecar_name
            sidecar_dst = Path(accepted_dir) / sidecar_name
            if sidecar_src.exists() and sidecar_name not in copied_sidecars and not sidecar_dst.exists():
                shutil.copy2(sidecar_src, sidecar_dst)
                copied_sidecars.add(sidecar_name)

        accepted_entries.append(
            {
                "batch": batch_idx,
                "source_file": str(path),
                "accepted_file": str(out_path),
                "energy_kj_per_mol": energy,
                "mol": mol,
            }
        )
    return accepted_entries


def write_batch_history(path, rows):
    fieldnames = [
        "batch",
        "seed",
        "generated_count",
        "relaxed_count",
        "negative_count",
        "accepted_total",
        "cluster_count",
        "cluster_gain",
        "low_gain_streak",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_accepted_summary(path, entries):
    fieldnames = ["batch", "source_file", "accepted_file", "energy_kj_per_mol"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "batch": entry["batch"],
                    "source_file": entry["source_file"],
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively generate and relax cap conformers, keep only negative-energy "
            "conformers, and stop when cluster growth saturates."
        )
    )
    parser.add_argument("--name", required=True, help="Cap name prefix.")
    parser.add_argument("--smiles", required=True, help="Cap SMILES.")
    parser.add_argument("--out", required=True, help="Output root directory.")
    parser.add_argument("--batch-size", type=int, required=True, help="Conformers generated per batch.")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum number of batches.")
    parser.add_argument("--seed-start", type=int, default=42, help="First batch seed; each next batch uses +1.")
    parser.add_argument("--cluster-rmsd", type=float, default=0.5, help="RMSD threshold for Butina clustering.")
    parser.add_argument(
        "--min-cluster-increase",
        type=int,
        default=5,
        help="Minimum cluster increase per batch before counting as saturation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop when cluster increase < min-cluster-increase for this many consecutive batches.",
    )
    parser.add_argument(
        "--schrodinger-run",
        default=None,
        help="Path to Schrodinger run launcher. Default: $SCHRODINGER/run",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for CappingGen.py.")
    parser.add_argument("-p", "--processors", type=int, default=1, help="Processors for relax_light.py.")
    parser.add_argument(
        "--cluster-processors",
        type=int,
        default=None,
        help="Processes for RMSD clustering. Default: same as --processors.",
    )
    parser.add_argument("--ligand-asl", default='chain.name "A"', help="Ligand ASL for relax_light.py.")
    parser.add_argument("--hydrogen-scope", choices=["all", "ligand"], default="all")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--grad-conv", type=float, default=0.05)
    parser.add_argument("--dielectric", type=float, default=80.0)
    parser.add_argument("--keep-batches", action="store_true", help="Keep generated/relaxed files for each batch.")
    args = parser.parse_args()
    if args.processors < 1:
        parser.error("--processors must be >= 1")
    if args.cluster_processors is not None and args.cluster_processors < 1:
        parser.error("--cluster-processors must be >= 1")
    return args


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    capping_gen_script = script_dir / "CappingGen.py"
    relax_light_script = script_dir / "relax_light.py"
    schrodinger_run = build_schrodinger_run_path(args.schrodinger_run)
    cluster_processes = args.cluster_processors or args.processors

    output_root = Path(args.out).expanduser().resolve()
    accepted_dir = output_root / "accepted_negative_sdf"
    batches_dir = output_root / "batches"
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    accepted_entries = []
    batch_rows = []
    low_gain_streak = 0
    previous_cluster_count = 0
    cluster_distances = []
    clustered_mol_count = 0

    for batch_idx in range(1, args.max_batches + 1):
        seed = args.seed_start + batch_idx - 1
        batch_name = f"{args.name}_b{batch_idx:04d}"
        batch_dir = batches_dir / f"batch_{batch_idx:04d}"
        generated_dir = batch_dir / "generated"
        relaxed_dir = batch_dir / "relaxed"
        generated_dir.mkdir(parents=True, exist_ok=True)
        relaxed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Batch {batch_idx} ===")
        print(f"Seed: {seed}")
        print("Step 1/4: generating conformers")
        run_cmd(
            [
                args.python,
                str(capping_gen_script),
                "--name",
                batch_name,
                "--smiles",
                args.smiles,
                "--num",
                str(args.batch_size),
                "--out",
                str(generated_dir),
                "--seed",
                str(seed),
            ]
        )

        print("Step 2/4: relaxing conformers")
        run_cmd(
            [
                str(schrodinger_run),
                "python3",
                str(relax_light_script),
                "-i",
                str(generated_dir),
                "-o",
                str(relaxed_dir),
                "--ligand-asl",
                args.ligand_asl,
                "--hydrogen-scope",
                args.hydrogen_scope,
                "--max-steps",
                str(args.max_steps),
                "--grad-conv",
                str(args.grad_conv),
                "--dielectric",
                str(args.dielectric),
                "-p",
                str(args.processors),
            ]
        )

        print("Step 3/4: keeping only negative-energy conformers")
        batch_accepted = filter_negative_relaxed_sdfs(relaxed_dir, accepted_dir, batch_idx)
        accepted_entries.extend(batch_accepted)

        print(f"Step 4/4: clustering accepted conformers ({cluster_processes} process(es))")
        all_mols = [entry["mol"] for entry in accepted_entries]
        validate_same_topology(all_mols)
        new_distances = pairwise_rmsd_distances(
            all_mols,
            start_i=clustered_mol_count,
            processes=cluster_processes,
        )
        cluster_distances.extend(new_distances)
        clustered_mol_count = len(all_mols)
        cluster_count = cluster_count_from_distances(
            cluster_distances,
            clustered_mol_count,
            args.cluster_rmsd,
        )
        cluster_gain = cluster_count - previous_cluster_count
        previous_cluster_count = cluster_count

        if cluster_gain < args.min_cluster_increase:
            low_gain_streak += 1
        else:
            low_gain_streak = 0

        generated_count = len(list(generated_dir.glob("*.sdf"))) + len(list(generated_dir.glob("*.sd")))
        relaxed_count = len(collect_relaxed_sdfs(relaxed_dir))
        row = {
            "batch": batch_idx,
            "seed": seed,
            "generated_count": generated_count,
            "relaxed_count": relaxed_count,
            "negative_count": len(batch_accepted),
            "accepted_total": len(accepted_entries),
            "cluster_count": cluster_count,
            "cluster_gain": cluster_gain,
            "low_gain_streak": low_gain_streak,
        }
        batch_rows.append(row)
        print(
            f"Generated: {generated_count}, relaxed: {relaxed_count}, negative: {len(batch_accepted)}, "
            f"clusters(total): {cluster_count}, cluster gain: {cluster_gain}, low-gain streak: {low_gain_streak}"
        )

        if not args.keep_batches:
            shutil.rmtree(batch_dir, ignore_errors=True)

        if low_gain_streak >= args.patience:
            print(
                f"Stopping early: cluster increase < {args.min_cluster_increase} "
                f"for {args.patience} consecutive batches."
            )
            break

    batch_history_csv = output_root / "batch_history.csv"
    accepted_csv = output_root / "accepted_negative_conformers.csv"
    write_batch_history(batch_history_csv, batch_rows)
    write_accepted_summary(accepted_csv, accepted_entries)

    print("\n=== Auto Conformer Summary ===")
    print(f"Accepted negative conformers: {len(accepted_entries)}")
    if batch_rows:
        print(f"Final cluster count: {batch_rows[-1]['cluster_count']}")
        print(f"Batches run: {batch_rows[-1]['batch']}")
    print(f"Accepted SDF directory: {accepted_dir}")
    print(f"Batch history CSV: {batch_history_csv}")
    print(f"Accepted conformer CSV: {accepted_csv}")


if __name__ == "__main__":
    main()
