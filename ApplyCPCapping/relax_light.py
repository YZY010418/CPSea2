#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from schrodinger import structure
from schrodinger.structutils import analyze, build, minimize

'''
run this script by $SCHRODINGER/run python3 this_script.py
'''


def strip_structure_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".pdb.gz"):
        return name[:-7]
    if name.endswith(".pdb"):
        return name[:-4]
    return path.stem


def collect_pdb_files(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith("_relaxed.pdb") or name.endswith("_relaxed.pdb.gz"):
            continue
        if name.endswith(".pdb") or name.endswith(".pdb.gz"):
            files.append(p)
    return files


def read_first_structure(path: Path):
    with structure.StructureReader(str(path)) as reader:
        try:
            st = next(reader)
        except StopIteration:
            raise ValueError(f"No structure found in file: {path}")
    return st


def add_hydrogens_to_structure(st, ligand_asl: str, scope: str):
    if scope == "all":
        build.add_hydrogens(st)
    elif scope == "ligand":
        ligand_atoms_initial = analyze.evaluate_asl(st, ligand_asl)
        if not ligand_atoms_initial:
            raise ValueError(f'No atoms matched ligand ASL before hydrogen addition: {ligand_asl}')
        build.add_hydrogens(st, atom_list=ligand_atoms_initial)
    else:
        raise ValueError(f"Unsupported hydrogen scope: {scope}")


def minimize_ligand_with_frozen_environment(
    st,
    ligand_asl: str,
    max_steps: int,
    grad_conv: float,
    dielectric_constant: float,
    verbose: bool = False,
):
    ligand_atoms = set(analyze.evaluate_asl(st, ligand_asl))
    if not ligand_atoms:
        raise ValueError(f'No atoms matched ligand ASL after hydrogen addition: {ligand_asl}')

    mz = minimize.Minimizer(
        struct=st,
        cleanup=False,
        max_steps=max_steps,
        gradient_criterion=grad_conv,
        dielectric_constant=dielectric_constant,
        verbose=verbose,
    )

    for atom in st.atom:
        if atom.index not in ligand_atoms:
            mz.addPosFrozen(atom.index)

    mz.minimize()
    relaxed_st = mz.getStructure()
    mz.cleanup_resources()
    return relaxed_st


def relax_one_structure(
    input_pdb: Path,
    output_dir: Path,
    ligand_asl: str,
    hydrogen_scope: str,
    max_steps: int,
    grad_conv: float,
    dielectric_constant: float,
    overwrite: bool,
    verbose: bool,
):
    stem = strip_structure_suffix(input_pdb)
    output_pdb = output_dir / f"{stem}_relaxed.pdb"

    if output_pdb.exists() and not overwrite:
        return "skip", output_pdb, 0.0

    start_time = time.time()
    st = read_first_structure(input_pdb)

    add_hydrogens_to_structure(st, ligand_asl=ligand_asl, scope=hydrogen_scope)

    relaxed_st = minimize_ligand_with_frozen_environment(
        st=st,
        ligand_asl=ligand_asl,
        max_steps=max_steps,
        grad_conv=grad_conv,
        dielectric_constant=dielectric_constant,
        verbose=verbose,
    )

    with structure.StructureWriter(str(output_pdb)) as writer:
        writer.append(relaxed_st)

    return "success", output_pdb, time.time() - start_time


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and minimize a directory of PDB/PDB.GZ structures using the "
            "Schrodinger Python API. Hydrogens are added, then only the ligand "
            "substructure is minimized while the remaining atoms are position-frozen."
        )
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Directory containing input .pdb or .pdb.gz files.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Directory for *_relaxed.pdb outputs.",
    )
    parser.add_argument(
        "--ligand-asl",
        default='chain.name "L"',
        help='ASL used to define the movable ligand atoms. Default: chain.name "L"',
    )
    parser.add_argument(
        "--hydrogen-scope",
        choices=["all", "ligand"],
        default="all",
        help=(
            "Where to add hydrogens before minimization. "
            "'all' is usually safer because the frozen receptor still contributes "
            "to the force field environment. Default: all"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum minimization steps. Default: 200",
    )
    parser.add_argument(
        "--grad-conv",
        type=float,
        default=0.05,
        help="Gradient convergence criterion. Smaller values are stricter. Default: 0.05",
    )
    parser.add_argument(
        "--dielectric",
        type=float,
        default=80.0,
        help="Constant dielectric used during minimization. Default: 80.0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_relaxed.pdb files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from the minimizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = collect_pdb_files(input_dir)

    if not pdb_files:
        print(f"No PDB or PDB.GZ files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} input structure files")
    print(f"Output directory: {output_dir}")
    print(f"Ligand ASL: {args.ligand_asl}")
    print(f"Hydrogen scope: {args.hydrogen_scope}")
    print(f"Max steps: {args.max_steps}")
    print(f"Gradient convergence: {args.grad_conv}")
    print(f"Dielectric constant: {args.dielectric}")

    results = []
    for pdb_file in tqdm(pdb_files, desc="Relaxing structures"):
        try:
            results.append(
                (pdb_file, *relax_one_structure(
                    input_pdb=pdb_file,
                    output_dir=output_dir,
                    ligand_asl=args.ligand_asl,
                    hydrogen_scope=args.hydrogen_scope,
                    max_steps=args.max_steps,
                    grad_conv=args.grad_conv,
                    dielectric_constant=args.dielectric,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                ))
            )
        except Exception as exc:
            results.append((pdb_file, "error", None, str(exc)))

    success_count = sum(1 for _, status, _, _ in results if status == "success")
    skip_count = sum(1 for _, status, _, _ in results if status == "skip")
    error_results = [(pdb_file, err) for pdb_file, status, _, err in results if status == "error"]

    print("\n=== Relaxation Results ===")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {len(error_results)}")

    if error_results:
        print("\nErrors:")
        for pdb_file, error in error_results[:20]:
            print(f"  {pdb_file}: {error}")


if __name__ == "__main__":
    main()
