#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import gzip
import multiprocessing as mp
import os
from pathlib import Path
import shlex
from typing import List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["RAYON_THREAD_LIMIT"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "UNK": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the receptor sequence from the first structure block in "
            "each .cif.gz file and write the results to a FASTA file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory containing .cif.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output FASTA path.",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def wrap_fasta_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def normalize_file_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".cif.gz"):
        return name[:-7]
    return path.stem


def collect_input_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*.cif.gz") if path.is_file())


def extract_first_data_block(lines: Sequence[str]) -> List[str]:
    block_start = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("data_"):
            block_start = idx
            break

    if block_start is None:
        return []

    block: List[str] = []
    for idx in range(block_start, len(lines)):
        line = lines[idx]
        if idx > block_start and line.lstrip().startswith("data_"):
            break
        block.append(line)
    return block


def find_atom_site_loop(block_lines: Sequence[str]) -> Optional[Tuple[List[str], int]]:
    idx = 0
    while idx < len(block_lines):
        stripped = block_lines[idx].strip()
        if stripped != "loop_":
            idx += 1
            continue

        idx += 1
        columns: List[str] = []
        while idx < len(block_lines):
            stripped = block_lines[idx].strip()
            if stripped.startswith("_"):
                columns.append(stripped)
                idx += 1
                continue
            break

        if columns and any(column.startswith("_atom_site.") for column in columns):
            return columns, idx

        while idx < len(block_lines):
            stripped = block_lines[idx].strip()
            if stripped == "#" or stripped == "loop_" or stripped.startswith("data_"):
                break
            idx += 1

    return None


def residue_code(res_name: str) -> str:
    clean_name = res_name.strip().upper()
    return AA3_TO_1.get(clean_name, "X")


def build_sequence_from_atom_site(columns: Sequence[str], block_lines: Sequence[str], start_idx: int) -> str:
    column_to_idx = {column: idx for idx, column in enumerate(columns)}

    comp_idx = column_to_idx.get("_atom_site.label_comp_id")
    if comp_idx is None:
        comp_idx = column_to_idx.get("_atom_site.auth_comp_id")
    if comp_idx is None:
        raise ValueError("No residue-name column found in _atom_site loop.")

    chain_idx = column_to_idx.get("_atom_site.auth_asym_id")
    if chain_idx is None:
        chain_idx = column_to_idx.get("_atom_site.label_asym_id")
    if chain_idx is None:
        raise ValueError("No chain-id column found in _atom_site loop.")

    seq_idx = column_to_idx.get("_atom_site.auth_seq_id")
    if seq_idx is None:
        seq_idx = column_to_idx.get("_atom_site.label_seq_id")
    if seq_idx is None:
        raise ValueError("No residue-index column found in _atom_site loop.")

    ins_idx = column_to_idx.get("_atom_site.pdbx_PDB_ins_code")
    group_idx = column_to_idx.get("_atom_site.group_PDB")

    seen_residues = set()
    sequence_chars: List[str] = []

    for idx in range(start_idx, len(block_lines)):
        stripped = block_lines[idx].strip()
        if not stripped:
            continue
        if stripped == "#":
            break
        if stripped == "loop_" or stripped.startswith("data_") or stripped.startswith("_"):
            break

        fields = shlex.split(block_lines[idx], posix=True)
        if len(fields) != len(columns):
            raise ValueError(
                f"Malformed _atom_site row with {len(fields)} values, expected {len(columns)}."
            )

        if group_idx is not None and fields[group_idx] != "ATOM":
            continue

        chain_id = fields[chain_idx]
        seq_id = fields[seq_idx]
        ins_code = "." if ins_idx is None else fields[ins_idx]
        residue_key = (chain_id, seq_id, ins_code)

        if residue_key in seen_residues:
            continue

        seen_residues.add(residue_key)
        sequence_chars.append(residue_code(fields[comp_idx]))

    return "".join(sequence_chars)


def extract_sequence_from_cif_gz(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    fasta_id = normalize_file_stem(path)

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()

        first_block = extract_first_data_block(lines)
        if not first_block:
            return fasta_id, None, "No data block found."

        atom_site_loop = find_atom_site_loop(first_block)
        if atom_site_loop is None:
            return fasta_id, None, "No _atom_site loop found in first data block."

        columns, start_idx = atom_site_loop
        sequence = build_sequence_from_atom_site(columns, first_block, start_idx)
        if not sequence:
            return fasta_id, None, "First data block contains no polymer ATOM residues."

        return fasta_id, sequence, None
    except Exception as exc:
        return fasta_id, None, str(exc)


def choose_chunksize(num_files: int, num_cpus: int) -> int:
    if num_files <= 0:
        return 1
    return max(1, min(256, num_files // max(1, num_cpus * 8) or 1))


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    num_cpus = max(1, args.cpus)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    cif_paths = collect_input_files(input_dir)
    if not cif_paths:
        raise SystemExit(f"No .cif.gz files found under: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.tsv")

    chunksize = choose_chunksize(len(cif_paths), num_cpus)
    extracted_count = 0
    failed_count = 0

    with mp.Pool(processes=num_cpus) as pool:
        iterator = pool.imap_unordered(extract_sequence_from_cif_gz, cif_paths, chunksize=chunksize)
        with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024) as fasta_handle, open(
            error_path, "w", encoding="utf-8", buffering=1024 * 1024
        ) as error_handle:
            for fasta_id, sequence, error in tqdm(
                iterator,
                total=len(cif_paths),
                desc="Extracting receptor sequences",
                unit="file",
            ):
                if error is not None:
                    failed_count += 1
                    error_handle.write(f"{fasta_id}\t{error}\n")
                    continue

                extracted_count += 1
                fasta_handle.write(f">{fasta_id}\n{wrap_fasta_sequence(sequence)}\n")

    print(f"Processed files: {len(cif_paths)}")
    print(f"FASTA entries written: {extracted_count}")
    print(f"Failed files: {failed_count}")
    print(f"Output FASTA: {output_path}")
    print(f"Error log: {error_path}")


if __name__ == "__main__":
    main()
