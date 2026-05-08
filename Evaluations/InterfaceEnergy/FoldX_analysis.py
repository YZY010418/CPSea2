#!/usr/bin/env python3
"""Unified FoldX entrypoint for PDB complexes and CPSea2 mmCIF files."""

import argparse
import time

DEFAULT_FOLDX_BIN = "~/FoldX/foldx_20270131"


def default_output(input_type):
    if input_type in {"cpsea2", "cif"}:
        return "foldx_cpsea2_interaction.csv"
    return "foldx_pdb_interaction.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FoldX Optimize/RepairPDB followed by AnalyseComplex for either "
            "PDB complexes or CPSea2 mmCIF files."
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
    parser.add_argument("--foldx", default=DEFAULT_FOLDX_BIN, help="Path to FoldX binary.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--seed", type=int, default=None, help="CPSea2 only: deterministic ligand selection seed.")
    parser.add_argument("--ligand_chain", default="L", help="Ligand chain ID.")
    parser.add_argument("--receptor_chain", default="R", help="Receptor chain ID.")
    parser.add_argument("--no_analyseComplexChains", action="store_true", help="Do not pass --analyseComplexChains.")
    parser.add_argument("--repair", action="store_true", help="Run RepairPDB instead of Optimize before AnalyseComplex.")
    parser.add_argument("--tmp_root", default=None, help="Temporary working directory root.")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary FoldX working directories.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each FoldX command.")
    parser.add_argument("--include_counts", action="store_true", help="PDB only: output ligand/receptor residue counts.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_type = "pdb" if args.input_type == "cpsea" else args.input_type
    output = args.output or default_output(input_type)
    start_time = time.time()

    if len(args.ligand_chain) != 1 or len(args.receptor_chain) != 1:
        raise ValueError("ligand_chain and receptor_chain must each be a single-character chain ID.")

    if input_type in {"cpsea2", "cif"}:
        import FoldX_analysis_CPSea2 as cpsea2_mode

        cpsea2_mode.process_cif_dir(
            input_dir=args.input,
            output_file=output,
            foldx_bin=args.foldx,
            n_jobs=args.jobs,
            seed=args.seed,
            recursive=args.recursive,
            ligand_chain=args.ligand_chain,
            receptor_chain=args.receptor_chain,
            use_analyse_complex_chains=not args.no_analyseComplexChains,
            use_repair=args.repair,
            tmp_root=args.tmp_root or "./tmp_foldx_cpsea2_analyse_complex",
            keep_tmp=args.keep_tmp,
            timeout=args.timeout,
        )
    else:
        import FoldX_analysis_CPSea as pdb_mode

        pdb_mode.process_pdb_dir(
            input_dir=args.input,
            output_file=output,
            foldx_bin=args.foldx,
            n_jobs=args.jobs,
            recursive=args.recursive,
            ligand_chain=args.ligand_chain,
            receptor_chain=args.receptor_chain,
            use_analyse_complex_chains=not args.no_analyseComplexChains,
            use_repair=args.repair,
            tmp_root=args.tmp_root or "./tmp_foldx_optimize_analyse_complex",
            keep_tmp=args.keep_tmp,
            timeout=args.timeout,
            include_counts=args.include_counts,
        )

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
