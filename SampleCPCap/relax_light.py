#!/usr/bin/env python3
import argparse
import inspect
import shutil
import subprocess
import sys
import time
from os import cpu_count
from pathlib import Path

class _TqdmFallback:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, n=1):
        pass

    def close(self):
        pass


def _multiprocessing_lock_available():
    try:
        from multiprocessing import RLock
        RLock()
        return True
    except Exception:
        return False


try:
    from tqdm import tqdm as _real_tqdm
except ImportError:
    _real_tqdm = None

if _real_tqdm is None or not _multiprocessing_lock_available():
    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable, **kwargs)
else:
    def tqdm(iterable=None, **kwargs):
        return _real_tqdm(iterable, **kwargs)

from schrodinger import structure
from schrodinger.structutils import analyze, build, minimize

'''
run this script by $SCHRODINGER/run python3 this_script.py
'''

SUPPORTED_INPUT_SUFFIXES = (
    ".pdb",
    ".pdb.gz",
    ".sdf",
    ".sd",
)

FORCE_FIELD_VALUE_ALIASES = {
    "OPLS4": ("OPLS4", "S-OPLS"),
    "S-OPLS": ("S-OPLS", "OPLS4"),
}
FORCE_FIELD_KEYWORD_CANDIDATES = (
    "force_field",
    "forcefield",
    "forcefield_version",
    "ffld_version",
    "opls_version",
)
PROGRESS_PRINT_INTERVAL = 1000


def print_completed_progress(done, total, force=False):
    if done <= 0:
        return
    if not force and done % PROGRESS_PRINT_INTERVAL != 0:
        return
    print(f"Done {done}/{total} files", file=sys.stderr, flush=True)


def force_field_values(force_field: str):
    value = str(force_field).strip()
    return FORCE_FIELD_VALUE_ALIASES.get(value.upper(), (value,))


def minimizer_signature_info():
    try:
        return inspect.signature(minimize.Minimizer)
    except (TypeError, ValueError):
        try:
            return inspect.signature(minimize.Minimizer.__init__)
        except (TypeError, ValueError):
            return None


def force_field_keyword_candidates(force_field: str):
    signature = minimizer_signature_info()
    if signature is None:
        return FORCE_FIELD_KEYWORD_CANDIDATES

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return FORCE_FIELD_KEYWORD_CANDIDATES

    candidates = [
        keyword for keyword in FORCE_FIELD_KEYWORD_CANDIDATES
        if keyword in parameters
    ]
    if candidates:
        return candidates

    accepted = ", ".join(parameters)
    raise RuntimeError(
        f"Cannot explicitly set force field {force_field!r}: "
        f"schrodinger.structutils.minimize.Minimizer does not expose a known "
        f"force-field keyword. Accepted parameters: {accepted}"
    )


def make_minimizer(base_kwargs, force_field: str):
    if not force_field:
        return minimize.Minimizer(**base_kwargs)

    errors = []
    for value in force_field_values(force_field):
        for keyword in force_field_keyword_candidates(value):
            try:
                kwargs = dict(base_kwargs)
                kwargs[keyword] = value
                return minimize.Minimizer(**kwargs)
            except Exception as exc:
                errors.append(f"{keyword}={value}: {exc}")

    raise RuntimeError(
        f"Could not create a Minimizer with explicit force field {force_field!r}. "
        f"Tried: {'; '.join(errors)}"
    )


def strip_structure_suffix(path: Path) -> str:
    name = path.name
    for suffix in (
        ".pdb.gz",
        ".pdb",
        ".sdf",
        ".sd",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def collect_structure_files(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "_relaxed." in name:
            continue
        if name.endswith(SUPPORTED_INPUT_SUFFIXES):
            files.append(p)
    return files


def output_suffix_for_input(input_structure: Path):
    name = input_structure.name.lower()
    if name.endswith((".sdf", ".sd")):
        return "sdf"
    if name.endswith((".pdb", ".pdb.gz")):
        return "pdb"
    raise ValueError(f"Unsupported input structure type: {input_structure}")


def expected_output_path(input_structure: Path, output_dir: Path):
    stem = strip_structure_suffix(input_structure)
    return output_dir / f"{stem}_relaxed.{output_suffix_for_input(input_structure)}"


def atom_name_sidecar_candidates(input_structure: Path):
    stem = strip_structure_suffix(input_structure)
    candidates = [input_structure.parent / f"{stem}_atom_names.csv"]
    if "_conformers_" in stem:
        cap_name = stem.split("_conformers_", 1)[0]
        candidates.append(input_structure.parent / f"{cap_name}_atom_names.csv")
    return candidates


def copy_atom_name_sidecar(input_structure: Path, output_dir: Path):
    for sidecar in atom_name_sidecar_candidates(input_structure):
        if not sidecar.exists():
            continue
        target = output_dir / sidecar.name
        if sidecar.resolve() != target.resolve():
            shutil.copy2(sidecar, target)
        return target
    return None


def read_first_structure(path: Path):
    with structure.StructureReader(str(path)) as reader:
        try:
            st = next(reader)
        except StopIteration:
            raise ValueError(f"No structure found in file: {path}")
    return st


def is_sdf_path(path: Path):
    return path.name.lower().endswith((".sdf", ".sd"))


def ligand_atom_indexes(st, ligand_asl: str, allow_all_fallback: bool = False):
    atoms = analyze.evaluate_asl(st, ligand_asl)
    if atoms or not allow_all_fallback:
        return atoms
    return [atom.index for atom in st.atom]


def add_hydrogens_to_structure(
    st,
    ligand_asl: str,
    scope: str,
    allow_all_ligand_fallback: bool = False,
):
    if scope == "all":
        build.add_hydrogens(st)
    elif scope == "ligand":
        ligand_atoms_initial = ligand_atom_indexes(
            st,
            ligand_asl,
            allow_all_fallback=allow_all_ligand_fallback,
        )
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
    force_field: str,
    verbose: bool = False,
    allow_all_ligand_fallback: bool = False,
):
    ligand_atoms = set(
        ligand_atom_indexes(
            st,
            ligand_asl,
            allow_all_fallback=allow_all_ligand_fallback,
        )
    )
    if not ligand_atoms:
        raise ValueError(f'No atoms matched ligand ASL after hydrogen addition: {ligand_asl}')

    minimizer_kwargs = {
        "struct": st,
        "cleanup": False,
        "max_steps": max_steps,
        "gradient_criterion": grad_conv,
        "dielectric_constant": dielectric_constant,
        "verbose": verbose,
    }
    mz = make_minimizer(minimizer_kwargs, force_field)

    for atom in st.atom:
        if atom.index not in ligand_atoms:
            mz.addPosFrozen(atom.index)

    mz.minimize()
    relaxed_st = mz.getStructure()
    mz.cleanup_resources()
    return relaxed_st


def relax_one_structure(
    input_structure: Path,
    output_dir: Path,
    ligand_asl: str,
    hydrogen_scope: str,
    max_steps: int,
    grad_conv: float,
    dielectric_constant: float,
    force_field: str,
    overwrite: bool,
    verbose: bool,
):
    stem = strip_structure_suffix(input_structure)
    output_structure = expected_output_path(input_structure, output_dir)

    if output_structure.exists() and not overwrite:
        copy_atom_name_sidecar(input_structure, output_dir)
        return "skip", output_structure, 0.0

    start_time = time.time()
    st = read_first_structure(input_structure)
    allow_all_ligand_fallback = is_sdf_path(input_structure)

    add_hydrogens_to_structure(
        st,
        ligand_asl=ligand_asl,
        scope=hydrogen_scope,
        allow_all_ligand_fallback=allow_all_ligand_fallback,
    )

    relaxed_st = minimize_ligand_with_frozen_environment(
        st=st,
        ligand_asl=ligand_asl,
        max_steps=max_steps,
        grad_conv=grad_conv,
        dielectric_constant=dielectric_constant,
        force_field=force_field,
        verbose=verbose,
        allow_all_ligand_fallback=allow_all_ligand_fallback,
    )

    with structure.StructureWriter(str(output_structure)) as writer:
        writer.append(relaxed_st)

    copy_atom_name_sidecar(input_structure, output_dir)
    return "success", output_structure, time.time() - start_time


def relax_worker(task):
    (
        structure_file,
        output_dir,
        ligand_asl,
        hydrogen_scope,
        max_steps,
        grad_conv,
        dielectric_constant,
        force_field,
        overwrite,
        verbose,
    ) = task

    try:
        return (
            structure_file,
            *relax_one_structure(
                input_structure=structure_file,
                output_dir=output_dir,
                ligand_asl=ligand_asl,
                hydrogen_scope=hydrogen_scope,
                max_steps=max_steps,
                grad_conv=grad_conv,
                dielectric_constant=dielectric_constant,
                force_field=force_field,
                overwrite=overwrite,
                verbose=verbose,
            ),
        )
    except Exception as exc:
        return (structure_file, "error", None, str(exc))


def worker_command(task):
    (
        structure_file,
        output_dir,
        ligand_asl,
        hydrogen_scope,
        max_steps,
        grad_conv,
        dielectric_constant,
        force_field,
        overwrite,
        verbose,
    ) = task

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-input",
        str(structure_file),
        "-o",
        str(output_dir),
        "--ligand-asl",
        ligand_asl,
        "--hydrogen-scope",
        hydrogen_scope,
        "--max-steps",
        str(max_steps),
        "--grad-conv",
        str(grad_conv),
        "--dielectric",
        str(dielectric_constant),
        "--force-field",
        force_field,
    ]
    if overwrite:
        cmd.append("--overwrite")
    if verbose:
        cmd.append("--verbose")
    return cmd


def result_from_completed_process(task, completed):
    structure_file = task[0]
    output_dir = task[1]
    output_structure = expected_output_path(structure_file, output_dir)

    if completed.returncode == 0:
        lines = (completed.stdout or "").strip().splitlines()
        for raw_line in reversed(lines):
            fields = raw_line.split("\t")
            if fields and fields[0] in {"success", "skip"}:
                elapsed = 0.0
                if len(fields) >= 3:
                    try:
                        elapsed = float(fields[2])
                    except ValueError:
                        elapsed = 0.0
                return (structure_file, fields[0], output_structure, elapsed)
        return (structure_file, "success", output_structure, 0.0)

    message = (completed.stderr or completed.stdout or "").strip()
    if not message:
        message = f"worker exited with return code {completed.returncode}"
    return (structure_file, "error", None, message)


def run_subprocess_relaxation_tasks(tasks, processors):
    processors = processors if processors > 0 else (cpu_count() or 1)
    processors = max(1, min(processors, len(tasks)))

    pending = list(tasks)
    running = {}
    results = []
    progress = tqdm(total=len(tasks), desc="Relaxing structures", unit="file")

    try:
        while pending or running:
            while pending and len(running) < processors:
                task = pending.pop(0)
                output_structure = expected_output_path(task[0], task[1])
                if output_structure.exists() and not task[8]:
                    copy_atom_name_sidecar(task[0], task[1])
                    results.append((task[0], "skip", output_structure, 0.0))
                    progress.update(1)
                    print_completed_progress(len(results), len(tasks))
                    continue

                process = subprocess.Popen(
                    worker_command(task),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                running[process] = task

            if not running:
                continue

            time.sleep(0.2)
            for process, task in list(running.items()):
                if process.poll() is None:
                    continue
                completed = process.communicate()
                completed_process = subprocess.CompletedProcess(
                    args=process.args,
                    returncode=process.returncode,
                    stdout=completed[0],
                    stderr=completed[1],
                )
                results.append(result_from_completed_process(task, completed_process))
                del running[process]
                progress.update(1)
                print_completed_progress(len(results), len(tasks))
    finally:
        progress.close()
        for process in running:
            process.terminate()

    if len(results) % PROGRESS_PRINT_INTERVAL != 0:
        print_completed_progress(len(results), len(tasks), force=True)
    return results


def run_relaxation_tasks(tasks, processors):
    processors = processors if processors > 0 else (cpu_count() or 1)
    processors = max(1, min(processors, len(tasks)))

    if processors == 1:
        results = []
        progress = tqdm(total=len(tasks), desc="Relaxing structures", unit="file")
        try:
            for task in tasks:
                results.append(relax_worker(task))
                progress.update(1)
                print_completed_progress(len(results), len(tasks))
        finally:
            progress.close()

        if len(results) % PROGRESS_PRINT_INTERVAL != 0:
            print_completed_progress(len(results), len(tasks), force=True)
        return results

    return run_subprocess_relaxation_tasks(tasks, processors)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and minimize a directory of SDF cap conformers or PDB merged "
            "complexes using the "
            "Schrodinger Python API. Hydrogens are added, then only the ligand "
            "substructure is minimized while the remaining atoms are position-frozen. "
            "SDF inputs are written as SDF; PDB inputs are written as PDB."
        )
    )
    parser.add_argument(
        "-i", "--input",
        required=False,
        type=Path,
        help="Directory containing input .sdf/.sd caps or .pdb/.pdb.gz merged complexes.",
    )
    parser.add_argument(
        "--single-input",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Directory for relaxed outputs.",
    )
    parser.add_argument(
        "--ligand-asl",
        default='chain.name "L"',
        help=(
            'ASL used to define movable ligand atoms. Default: chain.name "L". '
            "For SDF-only cap files, the whole molecule is used if this ASL matches nothing."
        ),
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
        "--force-field",
        default="OPLS4",
        help=(
            "Force field to request explicitly from Schrodinger Minimizer. "
            "Default: OPLS4. Use an empty string to use the installation default."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing relaxed output files.",
    )
    parser.add_argument(
        "-p",
        "--processors",
        type=int,
        default=1,
        help="Number of worker processes. Use 0 for all CPUs. Default: 1",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from the minimizer.",
    )
    args = parser.parse_args()
    if args.single_input is None and args.input is None:
        parser.error("one of -i/--input or --single-input is required")
    return args


def main():
    args = parse_args()
    output_dir = args.output.expanduser().resolve()

    if args.single_input is not None:
        input_structure = args.single_input.expanduser().resolve()
        if not input_structure.is_file():
            raise FileNotFoundError(f"Input structure does not exist: {input_structure}")
        output_dir.mkdir(parents=True, exist_ok=True)
        status, output_structure, elapsed = relax_one_structure(
            input_structure=input_structure,
            output_dir=output_dir,
            ligand_asl=args.ligand_asl,
            hydrogen_scope=args.hydrogen_scope,
            max_steps=args.max_steps,
            grad_conv=args.grad_conv,
            dielectric_constant=args.dielectric,
            force_field=args.force_field,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        print(f"{status}\t{output_structure}\t{elapsed:.2f}")
        return

    input_dir = args.input.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_files = collect_structure_files(input_dir)

    if not structure_files:
        print(f"No supported SDF or PDB files found in {input_dir}")
        return

    print(f"Found {len(structure_files)} input structure files")
    print(f"Output directory: {output_dir}")
    print("Output type: inferred from input (.sdf/.sd -> .sdf, .pdb/.pdb.gz -> .pdb)")
    print(f"Ligand ASL: {args.ligand_asl}")
    print(f"Hydrogen scope: {args.hydrogen_scope}")
    print(f"Max steps: {args.max_steps}")
    print(f"Gradient convergence: {args.grad_conv}")
    print(f"Dielectric constant: {args.dielectric}")
    print(f"Force field: {args.force_field or 'installation default'}")
    requested_processors = args.processors if args.processors > 0 else (cpu_count() or 1)
    active_processors = max(1, min(requested_processors, len(structure_files)))
    print(f"Processors: {active_processors}")

    tasks = [
        (
            structure_file,
            output_dir,
            args.ligand_asl,
            args.hydrogen_scope,
            args.max_steps,
            args.grad_conv,
            args.dielectric,
            args.force_field,
            args.overwrite,
            args.verbose,
        )
        for structure_file in structure_files
    ]

    results = run_relaxation_tasks(tasks, args.processors)

    success_count = sum(1 for _, status, _, _ in results if status == "success")
    skip_count = sum(1 for _, status, _, _ in results if status == "skip")
    error_results = [
        (structure_file, err)
        for structure_file, status, _, err in results
        if status == "error"
    ]

    print("\n=== Relaxation Results ===")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {len(error_results)}")

    if error_results:
        print("\nErrors:")
        for structure_file, error in error_results[:20]:
            print(f"  {structure_file}: {error}")


if __name__ == "__main__":
    main()
