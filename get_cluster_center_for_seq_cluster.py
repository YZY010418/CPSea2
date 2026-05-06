#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import shutil
from pathlib import Path


PDB_SUFFIXES = [
    ".pdb",
    ".PDB",
    ".ent",
    ".pdb.gz",
    ".ent.gz",
]


def open_text(path):
    path = Path(path)
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_ids_from_fasta(fasta_path):
    """
    Read FASTA headers.

    Header example:
        >A_C_1ABC|receptor_chain=R|ligand_chain=L

    Extracted ID:
        A_C_1ABC
    """
    ids = set()

    with open_text(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                continue

            header = line[1:].strip()
            if not header:
                continue

            pdb_id = header.split("|")[0].strip()
            if pdb_id:
                ids.add(pdb_id)

    return ids


def find_pdb_file(input_dir, basename):
    """
    Find PDB file by basename.
    """
    input_dir = Path(input_dir)

    for suffix in PDB_SUFFIXES:
        path = input_dir / f"{basename}{suffix}"
        if path.exists():
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Copy PDB files whose basenames appear in a FASTA header."
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Input folder containing PDB files."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="FASTA file. For each header, the first field split by '|' is used as the PDB basename."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output folder for copied PDB files."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_ids = read_ids_from_fasta(args.fasta)

    n_ids = len(fasta_ids)
    n_copied = 0
    n_missing = 0

    missing_ids = []

    for basename in sorted(fasta_ids):
        pdb_path = find_pdb_file(input_dir, basename)

        if pdb_path is None:
            n_missing += 1
            missing_ids.append(basename)
            continue

        out_path = output_dir / pdb_path.name
        shutil.copy2(pdb_path, out_path)
        n_copied += 1

    print("Done.")
    print(f"Unique IDs in FASTA: {n_ids}")
    print(f"Copied PDB files: {n_copied}")
    print(f"Missing PDB files: {n_missing}")

    if missing_ids:
        print("[Warning] Missing examples:")
        for x in missing_ids[:20]:
            print(f"  {x}")
        if len(missing_ids) > 20:
            print(f"  ... {len(missing_ids) - 20} more")


if __name__ == "__main__":
    main()
