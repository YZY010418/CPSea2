#!/usr/bin/env python3
import os
import re
import csv
import gzip
import time
import uuid
import shutil
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


DEFAULT_FOLDX_BIN = os.path.expanduser("~/FoldX/foldx_20270131")


def get_base_name(path):
    base = os.path.basename(path)

    if base.endswith(".gz"):
        base = os.path.splitext(base)[0]

    if base.endswith(".pdb"):
        base = os.path.splitext(base)[0]

    return base


def open_text_auto(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def list_pdb_files(input_dir, recursive=False):
    input_dir = Path(input_dir)
    suffixes = (".pdb", ".pdb.gz")

    if recursive:
        files = []
        for root, _, names in os.walk(input_dir):
            for name in names:
                if name.endswith(suffixes):
                    files.append(str(Path(root) / name))
        return sorted(files)

    return sorted(
        str(p)
        for p in input_dir.iterdir()
        if p.is_file() and p.name.endswith(suffixes)
    )


def parse_chain_id_from_pdb_line(line):
    if len(line) >= 22:
        return line[21].strip()
    return ""


def parse_residue_key_from_pdb_line(line):
    chain_id = parse_chain_id_from_pdb_line(line)
    resseq = line[22:26].strip() if len(line) >= 26 else ""
    icode = line[26].strip() if len(line) >= 27 else ""
    resname = line[17:20].strip() if len(line) >= 20 else ""
    return chain_id, resseq, icode, resname


def count_chain_residues(pdb_path, ligand_chain="L", receptor_chain="R"):
    ligand_atoms = 0
    receptor_atoms = 0
    ligand_residues = set()
    receptor_residues = set()

    with open_text_auto(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            chain_id = parse_chain_id_from_pdb_line(line)

            if chain_id == ligand_chain:
                ligand_atoms += 1
                ligand_residues.add(parse_residue_key_from_pdb_line(line))
            elif chain_id == receptor_chain:
                receptor_atoms += 1
                receptor_residues.add(parse_residue_key_from_pdb_line(line))

    if ligand_atoms == 0:
        raise ValueError(f"No atoms found for ligand chain {ligand_chain}")
    if receptor_atoms == 0:
        raise ValueError(f"No atoms found for receptor chain {receptor_chain}")

    return len(ligand_residues), len(receptor_residues)


def copy_or_decompress_pdb_to_workdir(src_pdb, workdir):
    """
    FoldX is most stable when the PDB file is in the current working directory
    and --pdb receives only the basename.
    """
    base_name = get_base_name(src_pdb)
    dst_pdb = Path(workdir) / f"{base_name}.pdb"

    if str(src_pdb).endswith(".gz"):
        with gzip.open(src_pdb, "rt") as fin, open(dst_pdb, "w") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copyfile(src_pdb, dst_pdb)

    return base_name, dst_pdb


def run_foldx_command(cmd, workdir, timeout=None, err_prefix="FoldX command failed"):
    result = subprocess.run(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{err_prefix}: return code {result.returncode}: {msg}")

    return result.stdout, result.stderr


def find_optimized_pdb(workdir, input_pdb_path, before_pdbs):
    """
    FoldX Optimize output naming may vary across versions.
    This function robustly finds the newly generated optimized PDB.

    It prefers newly created PDB files containing 'Optim' in their name.
    """
    workdir = Path(workdir)
    input_pdb_path = Path(input_pdb_path)
    input_name = input_pdb_path.name
    input_stem = input_pdb_path.stem

    after_pdbs = set(workdir.glob("*.pdb"))
    new_pdbs = sorted(after_pdbs - before_pdbs)

    if not new_pdbs:
        # Fallback: search all PDBs except the original input.
        new_pdbs = sorted(
            p for p in after_pdbs
            if p.name != input_name
        )

    if not new_pdbs:
        raise RuntimeError("FoldX Optimize finished but no optimized PDB file was found.")

    # Common possible naming patterns.
    preferred_names = [
        f"{input_stem}_Optimized.pdb",
        f"{input_stem}_Optimize.pdb",
        f"Optimized_{input_stem}.pdb",
        f"Optimize_{input_stem}.pdb",
        f"{input_stem}_optimized.pdb",
        f"optimized_{input_stem}.pdb",
    ]

    for name in preferred_names:
        candidate = workdir / name
        if candidate.exists():
            return candidate

    optim_candidates = [
        p for p in new_pdbs
        if "optim" in p.name.lower()
    ]

    if len(optim_candidates) == 1:
        return optim_candidates[0]

    if len(optim_candidates) > 1:
        return max(optim_candidates, key=lambda p: p.stat().st_mtime)

    # If there is only one newly generated PDB, use it.
    if len(new_pdbs) == 1:
        return new_pdbs[0]

    # Last fallback: use newest non-original PDB.
    return max(new_pdbs, key=lambda p: p.stat().st_mtime)


def run_foldx_optimize_then_analyse_complex(
    foldx_bin,
    pdb_basename,
    workdir,
    ligand_chain="L",
    receptor_chain="R",
    use_analyse_complex_chains=True,
    timeout=None,
):
    """
    Run:
        FoldX Optimize
        FoldX AnalyseComplex on optimized PDB

    All FoldX output is captured and muted.
    """
    workdir = Path(workdir)
    input_pdb = workdir / f"{pdb_basename}.pdb"

    before_pdbs = set(workdir.glob("*.pdb"))

    optimize_cmd = [
        foldx_bin,
        "--command=Optimize",
        f"--pdb={input_pdb.name}",
    ]

    run_foldx_command(
        optimize_cmd,
        workdir=workdir,
        timeout=timeout,
        err_prefix="FoldX Optimize failed",
    )

    optimized_pdb = find_optimized_pdb(
        workdir=workdir,
        input_pdb_path=input_pdb,
        before_pdbs=before_pdbs,
    )

    analyse_cmd = [
        foldx_bin,
        "--command=AnalyseComplex",
        f"--pdb={optimized_pdb.name}",
    ]

    if use_analyse_complex_chains:
        analyse_cmd.append(f"--analyseComplexChains={receptor_chain},{ligand_chain}")

    stdout_text, stderr_text = run_foldx_command(
        analyse_cmd,
        workdir=workdir,
        timeout=timeout,
        err_prefix="FoldX AnalyseComplex failed",
    )

    return stdout_text, stderr_text, optimized_pdb.name


def parse_interaction_total_from_stdout(stdout_text, receptor_chain="R", ligand_chain="L"):
    """
    Parse Total specifically under:

        interaction between R and L
        ...
        Total = -10.34

    Fallback also tries interaction between L and R.
    """
    pairs_to_try = [
        (receptor_chain, ligand_chain),
        (ligand_chain, receptor_chain),
    ]

    for chain_a, chain_b in pairs_to_try:
        header_pattern = (
            r"interaction\s+between\s+"
            + re.escape(chain_a)
            + r"\s+and\s+"
            + re.escape(chain_b)
        )

        header_match = re.search(header_pattern, stdout_text, flags=re.IGNORECASE)

        if not header_match:
            continue

        block = stdout_text[header_match.end():]

        stop_patterns = [
            r"\n\s*Complex Analysis of ",
            r"\n\s*Your file ",
            r"\n\s*End time of FoldX",
            r"\n\s*interaction\s+between\s+",
        ]

        stop_positions = []
        for pat in stop_patterns:
            m = re.search(pat, block, flags=re.IGNORECASE)
            if m:
                stop_positions.append(m.start())

        if stop_positions:
            block = block[: min(stop_positions)]

        total_matches = re.findall(
            r"^\s*Total\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$",
            block,
            flags=re.MULTILINE,
        )

        if total_matches:
            return float(total_matches[-1])

    return None


def parse_interaction_total_from_fxout(workdir, receptor_chain="R", ligand_chain="L"):
    """
    Conservative fallback parser for FoldX Interaction_*.fxout.
    Usually stdout parsing is enough.
    """
    workdir = Path(workdir)
    fxout_files = sorted(workdir.glob("Interaction_*.fxout"))

    for fxout in fxout_files:
        try:
            text = fxout.read_text(errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            has_r = re.search(
                rf"(^|[\s,;:_-]){re.escape(receptor_chain)}($|[\s,;:_-])",
                stripped,
            )
            has_l = re.search(
                rf"(^|[\s,;:_-]){re.escape(ligand_chain)}($|[\s,;:_-])",
                stripped,
            )

            if has_r and has_l:
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)
                if nums:
                    try:
                        return float(nums[-1])
                    except ValueError:
                        pass

    return None


def process_one_pdb(task):
    (
        pdb_path,
        foldx_bin,
        tmp_root,
        ligand_chain,
        receptor_chain,
        use_analyse_complex_chains,
        keep_tmp,
        timeout,
        include_counts,
    ) = task

    base_name = get_base_name(pdb_path)
    workdir = Path(tmp_root) / f"foldx_{base_name}_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if include_counts:
            n_ligand_res, n_receptor_res = count_chain_residues(
                pdb_path,
                ligand_chain=ligand_chain,
                receptor_chain=receptor_chain,
            )
        else:
            n_ligand_res, n_receptor_res = "", ""

        base_name, local_pdb = copy_or_decompress_pdb_to_workdir(
            src_pdb=pdb_path,
            workdir=workdir,
        )

        stdout_text, stderr_text, optimized_pdb_name = run_foldx_optimize_then_analyse_complex(
            foldx_bin=foldx_bin,
            pdb_basename=base_name,
            workdir=workdir,
            ligand_chain=ligand_chain,
            receptor_chain=receptor_chain,
            use_analyse_complex_chains=use_analyse_complex_chains,
            timeout=timeout,
        )

        score = parse_interaction_total_from_stdout(
            stdout_text,
            receptor_chain=receptor_chain,
            ligand_chain=ligand_chain,
        )

        if score is None:
            score = parse_interaction_total_from_fxout(
                workdir,
                receptor_chain=receptor_chain,
                ligand_chain=ligand_chain,
            )

        if score is None:
            raise RuntimeError(
                f"Could not parse interaction Total for {receptor_chain} and {ligand_chain}"
            )

        if include_counts:
            return [
                base_name,
                score,
                optimized_pdb_name,
                ligand_chain,
                receptor_chain,
                n_ligand_res,
                n_receptor_res,
                "",
            ]

        return [
            base_name,
            score,
            optimized_pdb_name,
            "",
        ]

    except Exception as e:
        err = str(e).replace("\n", " ")[:1000]

        if include_counts:
            return [
                base_name,
                "",
                "",
                ligand_chain,
                receptor_chain,
                "",
                "",
                err,
            ]

        return [
            base_name,
            "",
            "",
            err,
        ]

    finally:
        if not keep_tmp and workdir.exists():
            shutil.rmtree(workdir)


def process_pdb_dir(
    input_dir,
    output_file,
    foldx_bin=DEFAULT_FOLDX_BIN,
    n_jobs=1,
    recursive=False,
    ligand_chain="L",
    receptor_chain="R",
    use_analyse_complex_chains=True,
    tmp_root="./tmp_foldx_optimize_analyse_complex",
    keep_tmp=False,
    timeout=None,
    include_counts=False,
):
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    foldx_bin = os.path.expanduser(foldx_bin)

    if not os.path.exists(foldx_bin):
        raise FileNotFoundError(f"FoldX binary not found: {foldx_bin}")

    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    pdb_files = list_pdb_files(input_dir, recursive=recursive)
    total_files = len(pdb_files)

    print(f"Found {total_files} PDB files in {input_dir}")
    print(f"Using {n_jobs} worker processes")
    print(f"FoldX binary: {foldx_bin}")
    print(f"Pipeline: Optimize -> AnalyseComplex")
    print(f"Interaction: {receptor_chain} vs {ligand_chain}")
    print(f"Output CSV: {output_file}")

    tasks = [
        (
            pdb_path,
            foldx_bin,
            tmp_root,
            ligand_chain,
            receptor_chain,
            use_analyse_complex_chains,
            keep_tmp,
            timeout,
            include_counts,
        )
        for pdb_path in pdb_files
    ]

    with open(output_file, "w", newline="") as fh:
        writer = csv.writer(fh)

        if include_counts:
            writer.writerow(
                [
                    "input",
                    "foldx_interaction_total",
                    "optimized_pdb",
                    "ligand_chain",
                    "receptor_chain",
                    "n_ligand_residues",
                    "n_receptor_residues",
                    "error",
                ]
            )
        else:
            writer.writerow(
                [
                    "input",
                    "foldx_interaction_total",
                    "optimized_pdb",
                    "error",
                ]
            )

        if total_files == 0:
            return

        if n_jobs == 1:
            iterator = map(process_one_pdb, tasks)
            for row in tqdm(iterator, total=total_files, desc="Running FoldX", unit="pdb"):
                writer.writerow(row)
                fh.flush()
        else:
            with Pool(processes=n_jobs) as pool:
                for row in tqdm(
                    pool.imap_unordered(process_one_pdb, tasks, chunksize=1),
                    total=total_files,
                    desc="Running FoldX",
                    unit="pdb",
                ):
                    writer.writerow(row)
                    fh.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run FoldX Optimize followed by AnalyseComplex on .pdb/.pdb.gz complexes, "
            "and output only the interaction Total score between receptor chain R and ligand chain L."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing .pdb or .pdb.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="foldx_optimize_interaction_rl.csv",
        help="Output CSV file. Default: foldx_optimize_interaction_rl.csv",
    )
    parser.add_argument(
        "-p",
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes. Use -1 for all CPUs minus one. Default: 1.",
    )
    parser.add_argument(
        "--foldx",
        default=DEFAULT_FOLDX_BIN,
        help=f"Path to FoldX binary. Default: {DEFAULT_FOLDX_BIN}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .pdb/.pdb.gz files.",
    )
    parser.add_argument(
        "--ligand_chain",
        default="L",
        help="Ligand chain ID. Default: L.",
    )
    parser.add_argument(
        "--receptor_chain",
        default="R",
        help="Receptor chain ID. Default: R.",
    )
    parser.add_argument(
        "--no_analyseComplexChains",
        action="store_true",
        help=(
            "Do not pass --analyseComplexChains=R,L. "
            "Usually not recommended unless FoldX behaves unexpectedly."
        ),
    )
    parser.add_argument(
        "--tmp_root",
        default="./tmp_foldx_optimize_analyse_complex",
        help="Temporary working directory root. Default: ./tmp_foldx_optimize_analyse_complex",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary FoldX working directories for debugging.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each FoldX command. Default: no timeout.",
    )
    parser.add_argument(
        "--include_counts",
        action="store_true",
        help="Also output ligand/receptor residue counts.",
    )

    args = parser.parse_args()

    if len(args.ligand_chain) != 1 or len(args.receptor_chain) != 1:
        raise ValueError("ligand_chain and receptor_chain must each be a single-character chain ID.")

    start_time = time.time()

    process_pdb_dir(
        input_dir=args.input,
        output_file=args.output,
        foldx_bin=args.foldx,
        n_jobs=args.jobs,
        recursive=args.recursive,
        ligand_chain=args.ligand_chain,
        receptor_chain=args.receptor_chain,
        use_analyse_complex_chains=not args.no_analyseComplexChains,
        tmp_root=args.tmp_root,
        keep_tmp=args.keep_tmp,
        timeout=args.timeout,
        include_counts=args.include_counts,
    )

    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()