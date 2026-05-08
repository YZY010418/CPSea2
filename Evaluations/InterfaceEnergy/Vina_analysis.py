#!/usr/bin/env python3
"""Unified Vina entrypoint for PDB complexes and CPSea2 mmCIF files."""

import argparse
import time


def default_output(input_type):
    if input_type in {"cpsea2", "cif"}:
        return "check_vina_CPSea2_cif.csv"
    return "check_vina_pdb_lr.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Vina affinity for either PDB complexes with ligand/receptor "
            "chains or CPSea2 mmCIF files with mapped ligand blocks."
        )
    )
    parser.add_argument(
        "--input-type",
        choices=["pdb", "cpsea", "cpsea2", "cif"],
        required=True,
        help="Input layout: pdb/cpsea for .pdb(.gz), cpsea2/cif for CPSea2 .cif(.gz).",
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory.")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path.")
    parser.add_argument("-p", "--jobs", type=int, default=1, help="Worker processes; -1 uses CPUs minus one.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--score_only", action="store_true", help="Use Vina --score_only instead of --local_only.")
    parser.add_argument("--seed", type=int, default=None, help="CPSea2 only: deterministic ligand selection seed.")
    parser.add_argument("--ligand_chain", default="L", help="PDB only: ligand chain ID.")
    parser.add_argument("--receptor_chain", default="R", help="PDB only: receptor chain ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_type = "pdb" if args.input_type == "cpsea" else args.input_type
    output = args.output or default_output(input_type)
    start_time = time.time()

    if input_type in {"cpsea2", "cif"}:
        import Vina_analysis_CPSea2 as cpsea2_mode

        cpsea2_mode.process_cif_dir(
            input_dir=args.input,
            output_file=output,
            n_jobs=args.jobs,
            seed=args.seed,
            recursive=args.recursive,
            score_only=args.score_only,
        )
    else:
        import Vina_analysis_CPSea as pdb_mode

        if len(args.ligand_chain) != 1 or len(args.receptor_chain) != 1:
            raise ValueError("ligand_chain and receptor_chain must each be a single-character chain ID.")
        pdb_mode.process_pdb_dir(
            input_dir=args.input,
            output_file=output,
            n_jobs=args.jobs,
            recursive=args.recursive,
            score_only=args.score_only,
            ligand_chain=args.ligand_chain,
            receptor_chain=args.receptor_chain,
        )

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
