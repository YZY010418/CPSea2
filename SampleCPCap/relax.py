import argparse
import subprocess
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCHRODINGER = Path("/data_hdd/home/zhuwenyu/schrodinger2024-1/")


def run_command(cmd, cwd):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def remove_if_exists(path):
    if path.exists():
        path.unlink()


def write_prime_minimization_input(refine_inp, fixed_maegz, host):
    conf = f"""STRUCT_FILE\t{fixed_maegz.name}
JOB_TYPE\tREFINE
PRIME_TYPE\tREAL_MIN
SELECT\tasl = all
MINIM_NITER\t2
MINIM_NSTEP\t65
MINIM_RMSG\t0.010000
MINIM_METHOD\tau
USE_CRYSTAL_SYMMETRY\tno
USE_RANDOM_SEED\tno
SEED\t0
OPLS_VERSION S-OPLS
EXT_DIEL\t80.00
USE_MEMBRANE\tno
HOST\t{host}
"""
    refine_inp.write_text(conf)


def relax_pdb(input_pdb, output_dir, schrodinger, host="localhost:1", njobs=8, overwrite=False, keep_temp=False):
    input_pdb = input_pdb.resolve()
    output_dir = output_dir.resolve()
    stem = input_pdb.stem

    fixed_maegz = output_dir / f"{stem}_fixed.maegz"
    refine_inp = output_dir / f"{stem}_refine.inp"
    refine_out = output_dir / f"{stem}_refine-out.maegz"
    relaxed_pdb = output_dir / f"{stem}_relaxed.pdb"
    prepwizard_log = output_dir / f"prepwizard_{stem}.log"
    refine_log = output_dir / f"{stem}_refine.log"

    if relaxed_pdb.exists() and not overwrite:
        return "skip", relaxed_pdb, 0.0

    start_time = time.time()
    prepwizard = schrodinger / "utilities" / "prepwizard"
    prime = schrodinger / "prime"
    structconvert = schrodinger / "utilities" / "structconvert"

    run_command(
        [
            prepwizard,
            "-j",
            f"prepwizard_{stem}",
            "-watdist",
            "5",
            "-rehtreat",
            "-propka_pH",
            "7.4",
            "-HOST",
            host,
            "-noimpref",
            "-TMPLAUNCHDIR",
            "-ATTACHED",
            "-WAIT",
            input_pdb,
            fixed_maegz,
        ],
        cwd=output_dir,
    )

    write_prime_minimization_input(refine_inp, fixed_maegz, host)

    run_command([prime, refine_inp.name, "-NJOBS", str(njobs), "-WAIT"], cwd=output_dir)
    run_command([structconvert, refine_out.name, relaxed_pdb.name], cwd=output_dir)

    if not keep_temp:
        for temp_file in [fixed_maegz, refine_inp, refine_out, prepwizard_log, refine_log]:
            remove_if_exists(temp_file)

    return "success", relaxed_pdb, time.time() - start_time


def collect_pdb_files(input_dir):
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdb" and not p.stem.endswith("_relaxed")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and minimize PDB files with Schrodinger.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Directory containing raw PDB files.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Directory for *_relaxed.pdb files.")
    parser.add_argument("--schrodinger", type=Path, default=SCHRODINGER, help="Schrodinger installation path.")
    parser.add_argument("--host", default="localhost:1", help="Schrodinger HOST value.")
    parser.add_argument("--njobs", type=int, default=8, help="Prime -NJOBS value.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing *_relaxed.pdb outputs.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate Schrodinger files.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input.expanduser()
    output_dir = args.output.expanduser()
    schrodinger = args.schrodinger.expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = collect_pdb_files(input_dir)

    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files")
    print(f"Output directory: {output_dir}")

    results = []
    for pdb_file in tqdm(pdb_files, desc="Relaxing PDBs"):
        try:
            results.append(
                (pdb_file, *relax_pdb(
                    pdb_file,
                    output_dir,
                    schrodinger,
                    host=args.host,
                    njobs=args.njobs,
                    overwrite=args.overwrite,
                    keep_temp=args.keep_temp,
                ))
            )
        except subprocess.CalledProcessError as exc:
            results.append((pdb_file, "error", None, f"command failed with exit code {exc.returncode}"))
        except Exception as exc:
            results.append((pdb_file, "error", None, str(exc)))

    success_count = sum(1 for _, status, _, _ in results if status == "success")
    skip_count = sum(1 for _, status, _, _ in results if status == "skip")
    error_results = [(pdb, err) for pdb, status, _, err in results if status == "error"]

    print("\n=== Relaxation Results ===")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {len(error_results)}")

    if error_results:
        print("\nErrors:")
        for pdb_file, error in error_results[:10]:
            print(f"  {pdb_file}: {error}")


if __name__ == "__main__":
    main()
