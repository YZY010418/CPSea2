#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import os
import random
import re
import shutil
import time
import uuid
from multiprocessing import Pool, cpu_count
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from FoldX_analysis_CPSea import (
    DEFAULT_FOLDX_BIN,
    parse_interaction_total_from_fxout,
    parse_interaction_total_from_stdout,
    run_foldx_optimize_then_analyse_complex,
)


def normalize_ins_code(value):
    if value in (None, False, "", ".", "?", " "):
        return ""
    return str(value)


def ligand_sort_key(value):
    value = str(value)
    if value.isdigit():
        return 0, int(value)
    return 1, value


def get_cif_base_name(path):
    base = os.path.basename(path)

    if base.endswith(".gz"):
        base = os.path.splitext(base)[0]

    if base.endswith(".cif"):
        base = os.path.splitext(base)[0]

    return base


def atom_name_to_pdb_field(atom_name, element):
    atom_name = str(atom_name)
    element = ("" if element is None else str(element)).strip()

    if len(atom_name) >= 4:
        return atom_name[:4]

    if len(element) == 1 and atom_name and atom_name[0].isalpha():
        return f" {atom_name:<3}"

    return f"{atom_name:>4}"


def read_cif_document(path):
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "gemmi is required to read CPSea2 mmCIF files. Install it with "
            "`pip install gemmi` in the Python environment used for this script."
        ) from exc

    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            text = fh.read()
        return gemmi.cif.read_string(text)

    return gemmi.cif.read_file(str(path))


def choose_ligand_id(candidate_ids, base_name, seed=None):
    candidate_ids = sorted(candidate_ids, key=ligand_sort_key)

    if not candidate_ids:
        raise ValueError("No candidate ligand IDs found.")

    if seed is None:
        return random.choice(candidate_ids)

    digest = hashlib.md5(f"{seed}:{base_name}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(candidate_ids)
    return candidate_ids[idx]


def get_cif_value(category, column, row_idx, default=""):
    try:
        return category[column][row_idx]
    except Exception:
        return default


def write_atom_site_as_pdb_records(
    atom_site,
    fh,
    serial_start=1,
    allowed_residue_keys=None,
    chain_id_override=None,
):
    n_atoms = len(atom_site["id"])
    serial = serial_start
    written_atoms = 0
    residue_keys = set()
    output_resseq_by_key = {}

    for i in range(n_atoms):
        label_asym_id = str(get_cif_value(atom_site, "label_asym_id", i))
        label_seq_id = str(get_cif_value(atom_site, "label_seq_id", i))
        ins_code = normalize_ins_code(get_cif_value(atom_site, "pdbx_PDB_ins_code", i))
        input_res_key = (label_asym_id, label_seq_id, ins_code)

        if allowed_residue_keys is not None and input_res_key not in allowed_residue_keys:
            continue

        group_pdb = str(get_cif_value(atom_site, "group_PDB", i))
        record = "HETATM" if group_pdb.upper().startswith("HET") else "ATOM  "
        element = str(get_cif_value(atom_site, "type_symbol", i)).strip()
        atom_name = str(
            get_cif_value(
                atom_site,
                "auth_atom_id",
                i,
                default=get_cif_value(atom_site, "label_atom_id", i),
            )
        ).strip()
        res_name = str(
            get_cif_value(
                atom_site,
                "auth_comp_id",
                i,
                default=get_cif_value(atom_site, "label_comp_id", i),
            )
        ).strip()
        auth_chain_id = str(
            get_cif_value(
                atom_site,
                "auth_asym_id",
                i,
                default=label_asym_id,
            )
        ).strip()
        chain_id = chain_id_override or auth_chain_id or label_asym_id or "A"
        chain_id = chain_id[0]

        try:
            x = float(get_cif_value(atom_site, "Cartn_x", i))
            y = float(get_cif_value(atom_site, "Cartn_y", i))
            z = float(get_cif_value(atom_site, "Cartn_z", i))
        except Exception:
            continue

        # Some CPSea2 ligand caps share a sequence ID with the terminal residue.
        # Renumbering by residue identity keeps the temporary FoldX PDB valid.
        output_res_key = (label_asym_id, label_seq_id, ins_code, res_name)
        if output_res_key not in output_resseq_by_key:
            output_resseq_by_key[output_res_key] = len(output_resseq_by_key) + 1
        resseq = output_resseq_by_key[output_res_key]

        atom_name_field = atom_name_to_pdb_field(atom_name, element)
        altloc = " "
        icode = ins_code[:1] if ins_code else " "
        occupancy = 1.00
        temp_factor = 0.00

        line = (
            f"{record:<6}{serial:>5d} {atom_name_field}{altloc}{res_name:>3s} {chain_id:1s}"
            f"{resseq:>4d}{icode:1s}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occupancy:>6.2f}{temp_factor:>6.2f}          "
            f"{element:>2s}\n"
        )
        fh.write(line)
        residue_keys.add((chain_id, resseq, icode, res_name))
        serial += 1
        written_atoms += 1

    return written_atoms, len(residue_keys), serial


def find_cpsea2_blocks(doc):
    receptor_block = None
    ligand_blocks = {}

    for block in doc:
        name = block.name

        if name.endswith("_receptor") and receptor_block is None:
            receptor_block = block

        ligand_match = re.search(r"_ligand(\d+)$", name)
        if ligand_match:
            ligand_blocks[ligand_match.group(1)] = block

    if receptor_block is None and len(doc) > 0:
        receptor_block = doc[0]

    if not ligand_blocks:
        for block in doc[1:]:
            ligand_match = re.search(r"ligand(\d+)", block.name)
            if ligand_match:
                ligand_blocks[ligand_match.group(1)] = block

    if receptor_block is None:
        raise ValueError("No receptor block found in mmCIF file.")

    if not ligand_blocks:
        raise ValueError("No ligand blocks found in mmCIF file.")

    return receptor_block, ligand_blocks


def build_ligand_to_receptor_keys(receptor_block):
    mapping = receptor_block.get_mmcif_category("_cpsea_receptor_ligand_map.")

    if not mapping:
        raise ValueError("Missing _cpsea_receptor_ligand_map category in receptor block.")

    ligand_to_receptor_keys = {}
    n_rows = len(mapping["ligand_ids"])

    for i in range(n_rows):
        label_asym_id = str(mapping["label_asym_id"][i])
        label_seq_id = str(mapping["label_seq_id"][i])
        ins_code = normalize_ins_code(mapping["pdbx_PDB_ins_code"][i])
        res_key = (label_asym_id, label_seq_id, ins_code)
        ligand_ids_raw = str(mapping["ligand_ids"][i]).strip()

        if not ligand_ids_raw:
            continue

        ligand_ids = [x.strip() for x in ligand_ids_raw.split(",") if x.strip()]

        for ligand_id in ligand_ids:
            ligand_to_receptor_keys.setdefault(ligand_id, set()).add(res_key)

    return ligand_to_receptor_keys


def extract_receptor_ligand_complex(
    cif_path,
    workdir,
    seed=None,
    ligand_chain="L",
    receptor_chain="R",
):
    base_name = get_cif_base_name(cif_path)
    doc = read_cif_document(cif_path)

    if len(doc) < 2:
        raise ValueError("Expected at least one receptor block and one ligand block.")

    receptor_block, ligand_blocks = find_cpsea2_blocks(doc)
    ligand_to_receptor_keys = build_ligand_to_receptor_keys(receptor_block)

    candidate_ids = sorted(
        set(ligand_blocks.keys()) & set(ligand_to_receptor_keys.keys()),
        key=ligand_sort_key,
    )

    if not candidate_ids:
        raise ValueError("No overlapping ligand IDs between ligand blocks and receptor-ligand map.")

    ligand_id = choose_ligand_id(candidate_ids, base_name, seed=seed)
    receptor_keys = ligand_to_receptor_keys[ligand_id]
    ligand_block = ligand_blocks[ligand_id]

    receptor_atom_site = receptor_block.get_mmcif_category("_atom_site.")
    ligand_atom_site = ligand_block.get_mmcif_category("_atom_site.")

    if not receptor_atom_site or not ligand_atom_site:
        raise ValueError("Missing _atom_site category.")

    complex_base_name = f"{base_name}_lig{ligand_id}"
    complex_pdb = Path(workdir) / f"{complex_base_name}.pdb"

    with open(complex_pdb, "w") as fh:
        n_receptor_atoms, n_receptor_res, next_serial = write_atom_site_as_pdb_records(
            receptor_atom_site,
            fh,
            serial_start=1,
            allowed_residue_keys=receptor_keys,
            chain_id_override=receptor_chain,
        )

        if n_receptor_atoms > 0:
            fh.write("TER\n")

        n_ligand_atoms, n_ligand_res, _ = write_atom_site_as_pdb_records(
            ligand_atom_site,
            fh,
            serial_start=next_serial,
            allowed_residue_keys=None,
            chain_id_override=ligand_chain,
        )

        if n_ligand_atoms > 0:
            fh.write("TER\n")

        fh.write("END\n")

    if n_receptor_atoms == 0:
        raise ValueError(f"Chosen ligand {ligand_id} produced an empty receptor selection.")

    if n_ligand_atoms == 0:
        raise ValueError(f"Chosen ligand {ligand_id} produced an empty ligand.")

    return (
        base_name,
        ligand_id,
        complex_base_name,
        complex_pdb,
        len(candidate_ids),
        n_ligand_res,
        n_receptor_res,
    )


def process_one_cif(task):
    (
        cif_path,
        foldx_bin,
        tmp_root,
        seed,
        ligand_chain,
        receptor_chain,
        use_analyse_complex_chains,
        use_repair,
        keep_tmp,
        timeout,
    ) = task

    base_name = get_cif_base_name(cif_path)
    workdir = Path(tmp_root) / f"foldx_cpsea2_{base_name}_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        (
            base_name,
            ligand_id,
            complex_base_name,
            _complex_pdb,
            n_candidates,
            n_ligand_res,
            n_receptor_res,
        ) = extract_receptor_ligand_complex(
            cif_path=cif_path,
            workdir=workdir,
            seed=seed,
            ligand_chain=ligand_chain,
            receptor_chain=receptor_chain,
        )

        stdout_text, _stderr_text, prepared_pdb_name = run_foldx_optimize_then_analyse_complex(
            foldx_bin=foldx_bin,
            pdb_basename=complex_base_name,
            workdir=workdir,
            ligand_chain=ligand_chain,
            receptor_chain=receptor_chain,
            use_analyse_complex_chains=use_analyse_complex_chains,
            use_repair=use_repair,
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

        return [
            base_name,
            ligand_id,
            n_candidates,
            n_ligand_res,
            n_receptor_res,
            score,
            prepared_pdb_name,
            "",
        ]

    except Exception as e:
        err = str(e).replace("\n", " ")[:1000]
        return [base_name, "", "", "", "", "", "", err]

    finally:
        if not keep_tmp and workdir.exists():
            shutil.rmtree(workdir)


def list_cif_files(input_dir, recursive=False):
    input_dir = Path(input_dir)
    suffixes = (".cif", ".cif.gz")

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


def process_cif_dir(
    input_dir,
    output_file,
    foldx_bin=DEFAULT_FOLDX_BIN,
    n_jobs=1,
    seed=None,
    recursive=False,
    ligand_chain="L",
    receptor_chain="R",
    use_analyse_complex_chains=True,
    use_repair=False,
    tmp_root="./tmp_foldx_cpsea2_analyse_complex",
    keep_tmp=False,
    timeout=None,
):
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    foldx_bin = os.path.expanduser(foldx_bin)

    if not os.path.exists(foldx_bin):
        raise FileNotFoundError(f"FoldX binary not found: {foldx_bin}")

    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    cif_files = list_cif_files(input_dir, recursive=recursive)
    total_files = len(cif_files)

    print(f"Found {total_files} CIF files in {input_dir}")
    print(f"Using {n_jobs} worker processes")
    print(f"FoldX binary: {foldx_bin}")
    prepare_command = "RepairPDB" if use_repair else "Optimize"
    print(f"Pipeline: {prepare_command} -> AnalyseComplex")
    print(f"Interaction: {receptor_chain} vs {ligand_chain}")
    print(f"Output CSV: {output_file}")

    tasks = [
        (
            cif_path,
            foldx_bin,
            tmp_root,
            seed,
            ligand_chain,
            receptor_chain,
            use_analyse_complex_chains,
            use_repair,
            keep_tmp,
            timeout,
        )
        for cif_path in cif_files
    ]

    with open(output_file, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "input",
                "ligand_id",
                "n_candidate_ligands",
                "n_ligand_residues",
                "n_receptor_residues",
                "foldx_interaction_total",
                "prepared_pdb",
                "error",
            ]
        )

        if total_files == 0:
            return

        if n_jobs == 1:
            iterator = map(process_one_cif, tasks)
            for row in tqdm(iterator, total=total_files, desc="Running FoldX", unit="cif"):
                writer.writerow(row)
                fh.flush()
        else:
            with Pool(processes=n_jobs) as pool:
                for row in tqdm(
                    pool.imap_unordered(process_one_cif, tasks, chunksize=1),
                    total=total_files,
                    desc="Running FoldX",
                    unit="cif",
                ):
                    writer.writerow(row)
                    fh.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select one mapped ligand-receptor pair from each CPSea2 "
            "mmCIF(.gz) file, write a temporary R/L complex PDB, then run FoldX "
            "Optimize or RepairPDB followed by AnalyseComplex."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing .cif or .cif.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="foldx_cpsea2_interaction.csv",
        help="Output CSV file. Default: foldx_cpsea2_interaction.csv",
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
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic ligand selection per file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .cif/.cif.gz files under input directory.",
    )
    parser.add_argument(
        "--ligand_chain",
        default="L",
        help="Ligand chain ID to write into the temporary PDB. Default: L.",
    )
    parser.add_argument(
        "--receptor_chain",
        default="R",
        help="Receptor chain ID to write into the temporary PDB. Default: R.",
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
        "--repair",
        action="store_true",
        help="Run FoldX RepairPDB instead of Optimize before AnalyseComplex.",
    )
    parser.add_argument(
        "--tmp_root",
        default="./tmp_foldx_cpsea2_analyse_complex",
        help="Temporary working directory root. Default: ./tmp_foldx_cpsea2_analyse_complex",
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

    args = parser.parse_args()

    if len(args.ligand_chain) != 1 or len(args.receptor_chain) != 1:
        raise ValueError("ligand_chain and receptor_chain must each be a single-character chain ID.")

    start_time = time.time()

    process_cif_dir(
        input_dir=args.input,
        output_file=args.output,
        foldx_bin=args.foldx,
        n_jobs=args.jobs,
        seed=args.seed,
        recursive=args.recursive,
        ligand_chain=args.ligand_chain,
        receptor_chain=args.receptor_chain,
        use_analyse_complex_chains=not args.no_analyseComplexChains,
        use_repair=args.repair,
        tmp_root=args.tmp_root,
        keep_tmp=args.keep_tmp,
        timeout=args.timeout,
    )

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
