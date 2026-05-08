#!/usr/bin/env python3
"""Unified PLIP entrypoint for CPSea2/CPSea2AFDB mmCIFs and PepBench PDBs."""

import argparse
import os
import time


STRUCTURE_SUFFIXES = (
    ".cif.gz",
    ".mmcif.gz",
    ".pdb.gz",
    ".cif",
    ".mmcif",
    ".pdb",
    ".gz",
)


def strip_structure_suffix(path):
    name = os.path.basename(path)
    for suffix in STRUCTURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def normalize_sample_name(path, drop_suffix_fields=0, separator="_"):
    """Normalize names by removing structure suffixes, then optional trailing fields."""
    name = strip_structure_suffix(path)
    if drop_suffix_fields <= 0:
        return name

    fields = name.split(separator)
    if len(fields) <= drop_suffix_fields:
        return name
    return separator.join(fields[:-drop_suffix_fields])


def patch_name_parser(module, function_name, drop_suffix_fields, separator):
    if drop_suffix_fields <= 0:
        return

    def patched(path):
        return normalize_sample_name(path, drop_suffix_fields, separator)

    setattr(module, function_name, patched)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PLIP on CPSea2/CPSea2AFDB multi-block mmCIF files or PepBench "
            "PDB/PDB.GZ complexes. PepBDB peptide.pdb/receptor.pdb folders are not supported here."
        )
    )
    parser.add_argument(
        "--input-type",
        choices=["cpsea2", "cpsea2afdb", "pepbench"],
        required=True,
        help="Input layout to process.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV. For CPSea2/CPSea2AFDB this is an alias for --output-all.",
    )
    parser.add_argument(
        "--output-all",
        default=None,
        help="CPSea2/CPSea2AFDB only: PLIP results for all ligands in each mmCIF.",
    )
    parser.add_argument(
        "--output-random",
        default=None,
        help="CPSea2/CPSea2AFDB only: PLIP results for one random ligand per mmCIF.",
    )
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Worker processes; -1 uses CPUs minus one.")
    parser.add_argument("--log", default=None, help="Optional log file path.")
    parser.add_argument("--tmp-root", default="./tmp3", help="CPSea2/CPSea2AFDB only: temporary working directory.")
    parser.add_argument("--keep-complexes", action="store_true", help="CPSea2/CPSea2AFDB only: keep generated complex PDBs.")
    parser.add_argument("--complex-cache-dir", default=None, help="CPSea2/CPSea2AFDB only: directory for kept complex PDBs.")
    parser.add_argument("--recursive", action="store_true", help="CPSea2/CPSea2AFDB only: recursively search for CIF files.")
    parser.add_argument(
        "--ignore-map-chain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Ignore chain/asym IDs in the receptor-ligand map. By default this is "
            "enabled for cpsea2afdb and disabled for cpsea2."
        ),
    )
    parser.add_argument(
        "--map-ligand-index-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="CPSea2/CPSea2AFDB only: index base used by ligand ids in the receptor-ligand map.",
    )
    parser.add_argument(
        "--map-residue-offset",
        type=int,
        default=0,
        help="CPSea2/CPSea2AFDB only: offset added to residue numbers read from the receptor-ligand map.",
    )
    parser.add_argument(
        "--map-fallback",
        choices=["auto", "error", "whole-receptor"],
        default="auto",
        help="CPSea2/CPSea2AFDB only: fallback behavior when the receptor-ligand map is missing or unusable.",
    )
    parser.add_argument(
        "--allow-whole-receptor-fallback",
        action="store_true",
        help="CPSea2/CPSea2AFDB only: allow whole-receptor fallback on map matching failures.",
    )
    parser.add_argument(
        "--plip-bin",
        default="plip",
        help="PLIP executable path or command, for example 'plip' or 'python -m plip.plipcmd'.",
    )
    parser.add_argument(
        "--name-drop-fields",
        type=int,
        default=0,
        help=(
            "Drop this many trailing separator-delimited fields from output sample names "
            "after removing the file suffix. Useful for CPSea2AFDB names where variant "
            "tokens are appended at the end."
        ),
    )
    parser.add_argument(
        "--name-separator",
        default="_",
        help="Separator used by --name-drop-fields.",
    )
    return parser.parse_args()


def run_cpsea_mode(args):
    import plip_analysis_mp_cpsea2afdb as cpsea

    output_all = args.output_all or args.output
    ignore_map_chain = args.ignore_map_chain
    if ignore_map_chain is None:
        ignore_map_chain = args.input_type == "cpsea2afdb"

    patch_name_parser(
        cpsea,
        "strip_cif_gz_suffix",
        args.name_drop_fields,
        args.name_separator,
    )

    cpsea.process_input_folder(
        input_dir=args.input,
        output_all_file=output_all,
        output_random_file=args.output_random,
        n_jobs=args.jobs,
        log_file=args.log,
        tmp_root=args.tmp_root,
        keep_complexes=args.keep_complexes,
        complex_cache_dir=args.complex_cache_dir,
        recursive=args.recursive,
        ignore_map_chain=ignore_map_chain,
        map_ligand_index_base=args.map_ligand_index_base,
        map_residue_offset=args.map_residue_offset,
        allow_whole_receptor_fallback=args.allow_whole_receptor_fallback,
        map_fallback_policy=args.map_fallback,
        plip_bin=args.plip_bin,
    )


def run_pepbench_mode(args):
    import plip_analysis_mp_pepbench as pepbench

    patch_name_parser(
        pepbench,
        "get_sample_name",
        args.name_drop_fields,
        args.name_separator,
    )
    pepbench.process_input_folder(
        input_dir=args.input,
        output_file=args.output or "PLIP_metadata_PepBench.csv",
        n_jobs=args.jobs,
        log_file=args.log,
    )


def main():
    args = parse_args()
    start_time = time.time()

    if args.input_type in {"cpsea2", "cpsea2afdb"}:
        run_cpsea_mode(args)
    else:
        run_pepbench_mode(args)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
