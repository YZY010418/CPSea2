#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import gzip
import multiprocessing as mp
import os
from pathlib import Path
import shlex
from typing import Iterator, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


# Avoid hidden over-subscription when many worker processes are used.
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
            "Stream-extract receptor sequences from .cif or .cif.gz files and "
            "write them to a FASTA file. The parser reads each CIF line by line "
            "instead of loading the full file into memory."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory containing .cif or .cif.gz files.",
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
    parser.add_argument(
        "--model-id",
        default="1",
        help=(
            "Only extract residues from this _atom_site.pdbx_PDB_model_num value "
            "when the column exists. Default: 1. Use --all-models to disable."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Disable model filtering and extract unique residues across all atom_site rows. "
            "This is closer to the old behavior, but slower for multi-model CIF files."
        ),
    )
    parser.add_argument(
        "--unordered",
        action="store_true",
        help=(
            "Write FASTA entries as soon as workers finish. Faster, but output order "
            "is not guaranteed. Default preserves sorted file order."
        ),
    )
    parser.add_argument(
        "--no-scan-progress",
        action="store_true",
        help="Disable tqdm progress bar while scanning input files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort a file on malformed _atom_site rows. Default skips malformed rows.",
    )
    return parser.parse_args()


def wrap_fasta_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def normalize_file_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".cif.gz"):
        return name[:-7]
    if name.endswith(".cif"):
        return name[:-4]
    return path.stem


def iter_input_files(input_dir: Path) -> Iterator[Path]:
    for path in input_dir.rglob("*"):
        if path.is_file() and (path.name.endswith(".cif") or path.name.endswith(".cif.gz")):
            yield path


def collect_input_files(input_dir: Path, show_progress: bool = True) -> List[Path]:
    iterator = iter_input_files(input_dir)
    if show_progress:
        iterator = tqdm(iterator, desc="Scanning CIF files", unit="file")
    return sorted(iterator)


def open_cif_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def residue_code(res_name: str) -> str:
    clean_name = res_name.strip().upper()
    return AA3_TO_1.get(clean_name, "X")


def is_loop_boundary(stripped: str) -> bool:
    return (
        stripped == "#"
        or stripped == "loop_"
        or stripped.startswith("data_")
        or stripped.startswith("_")
    )


def parse_atom_site_row(line: str, expected_cols: int) -> Optional[List[str]]:
    try:
        fields = shlex.split(line, posix=True)
    except ValueError:
        return None

    if len(fields) != expected_cols:
        return None
    return fields


def get_required_col(
    column_to_idx: dict,
    candidates: Sequence[str],
    error_message: str,
) -> int:
    for name in candidates:
        idx = column_to_idx.get(name)
        if idx is not None:
            return idx
    raise ValueError(error_message)


def build_sequence_from_streamed_atom_site(
    handle,
    columns: Sequence[str],
    first_data_line: str,
    model_id: Optional[str],
    strict: bool,
) -> str:
    column_to_idx = {column: idx for idx, column in enumerate(columns)}

    comp_idx = get_required_col(
        column_to_idx,
        ("_atom_site.label_comp_id", "_atom_site.auth_comp_id"),
        "No residue-name column found in _atom_site loop.",
    )
    chain_idx = get_required_col(
        column_to_idx,
        ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"),
        "No chain-id column found in _atom_site loop.",
    )
    seq_idx = get_required_col(
        column_to_idx,
        ("_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
        "No residue-index column found in _atom_site loop.",
    )

    ins_idx = column_to_idx.get("_atom_site.pdbx_PDB_ins_code")
    group_idx = column_to_idx.get("_atom_site.group_PDB")
    model_idx = column_to_idx.get("_atom_site.pdbx_PDB_model_num")

    seen_residues = set()
    sequence_chars: List[str] = []
    target_model_seen = False

    def consume_data_line(raw_line: str) -> bool:
        """
        Return True if parsing should continue, False if the atom_site loop is done.
        """
        nonlocal target_model_seen

        stripped = raw_line.strip()
        if not stripped:
            return True

        if is_loop_boundary(stripped):
            return False

        fields = parse_atom_site_row(raw_line, len(columns))
        if fields is None:
            if strict:
                raise ValueError(
                    f"Malformed _atom_site row. Expected {len(columns)} columns: {raw_line[:200]}"
                )
            return True

        if group_idx is not None and fields[group_idx] != "ATOM":
            return True

        if model_id is not None and model_idx is not None:
            row_model = fields[model_idx]
            if row_model != model_id:
                # In mmCIF files, atom rows are normally grouped by model.
                # Once model 1 has already been read, stop early when model 2 begins.
                if target_model_seen:
                    return False
                return True
            target_model_seen = True

        chain_id = fields[chain_idx]
        seq_id = fields[seq_idx]
        ins_code = "." if ins_idx is None else fields[ins_idx]
        residue_key = (chain_id, seq_id, ins_code)

        if residue_key in seen_residues:
            return True

        seen_residues.add(residue_key)
        sequence_chars.append(residue_code(fields[comp_idx]))
        return True

    # The first non-column line has already been read by the caller.
    if first_data_line:
        if not consume_data_line(first_data_line):
            return "".join(sequence_chars)

    for raw_line in handle:
        if not consume_data_line(raw_line):
            break

    return "".join(sequence_chars)


def extract_sequence_from_cif_stream(
    path: Path,
    model_id: Optional[str] = "1",
    strict: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (fasta_id, sequence, error).

    This function intentionally streams the first data block and first _atom_site loop
    instead of materializing the entire CIF file in memory.
    """
    fasta_id = normalize_file_stem(path)

    try:
        with open_cif_text(path) as handle:
            in_first_data_block = False

            for raw_line in handle:
                stripped = raw_line.strip()

                if not in_first_data_block:
                    if stripped.startswith("data_"):
                        in_first_data_block = True
                    continue

                # Do not enter later data blocks.
                if stripped.startswith("data_"):
                    break

                if stripped != "loop_":
                    continue

                columns: List[str] = []
                first_data_line = ""

                # Read loop columns.
                for raw_col_line in handle:
                    col_stripped = raw_col_line.strip()

                    if col_stripped.startswith("data_"):
                        # Empty loop before a new block.
                        first_data_line = raw_col_line
                        break

                    if col_stripped.startswith("_"):
                        columns.append(col_stripped)
                        continue

                    first_data_line = raw_col_line
                    break

                if not columns:
                    continue

                is_atom_site_loop = any(column.startswith("_atom_site.") for column in columns)

                if not is_atom_site_loop:
                    # Skip this non-atom_site loop.
                    if first_data_line:
                        first_stripped = first_data_line.strip()
                        if first_stripped.startswith("data_"):
                            break
                        if first_stripped == "loop_":
                            # This rare case is intentionally not reprocessed because it
                            # would require a pushback buffer. Standard mmCIF loops end
                            # with '#', so this should not affect normal files.
                            continue
                        if first_stripped != "#" and not first_stripped.startswith("_"):
                            for raw_skip_line in handle:
                                skip_stripped = raw_skip_line.strip()
                                if is_loop_boundary(skip_stripped):
                                    break
                    continue

                sequence = build_sequence_from_streamed_atom_site(
                    handle=handle,
                    columns=columns,
                    first_data_line=first_data_line,
                    model_id=model_id,
                    strict=strict,
                )

                if not sequence:
                    model_msg = f" for model {model_id}" if model_id is not None else ""
                    return fasta_id, None, f"First _atom_site loop contains no polymer ATOM residues{model_msg}."

                return fasta_id, sequence, None

        return fasta_id, None, "No _atom_site loop found in first data block."
    except Exception as exc:
        return fasta_id, None, str(exc)


# Worker globals avoid repeatedly pickling options for each task.
_WORKER_MODEL_ID: Optional[str] = "1"
_WORKER_STRICT: bool = False


def init_worker(model_id: Optional[str], strict: bool) -> None:
    global _WORKER_MODEL_ID, _WORKER_STRICT
    _WORKER_MODEL_ID = model_id
    _WORKER_STRICT = strict


def extract_sequence_worker(path: Path) -> Tuple[str, Optional[str], Optional[str]]:
    return extract_sequence_from_cif_stream(
        path=path,
        model_id=_WORKER_MODEL_ID,
        strict=_WORKER_STRICT,
    )


def choose_chunksize(num_files: int, num_cpus: int) -> int:
    if num_files <= 0:
        return 1
    return max(1, min(256, num_files // max(1, num_cpus * 8) or 1))


def write_result(
    fasta_handle,
    error_handle,
    fasta_id: str,
    sequence: Optional[str],
    error: Optional[str],
) -> Tuple[int, int]:
    if error is not None:
        error_handle.write(f"{fasta_id}\t{error}\n")
        return 0, 1

    assert sequence is not None
    fasta_handle.write(f">{fasta_id}\n{wrap_fasta_sequence(sequence)}\n")
    return 1, 0


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    num_cpus = max(1, args.cpus)
    model_id = None if args.all_models else str(args.model_id)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    cif_paths = collect_input_files(
        input_dir,
        show_progress=not args.no_scan_progress,
    )
    if not cif_paths:
        raise SystemExit(f"No .cif or .cif.gz files found under: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.tsv")

    chunksize = choose_chunksize(len(cif_paths), num_cpus)
    extracted_count = 0
    failed_count = 0

    with mp.Pool(
        processes=num_cpus,
        initializer=init_worker,
        initargs=(model_id, args.strict),
    ) as pool:
        map_func = pool.imap_unordered if args.unordered else pool.imap
        iterator = map_func(extract_sequence_worker, cif_paths, chunksize=chunksize)

        with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as fasta_handle, open(
            error_path, "w", encoding="utf-8", buffering=1024 * 1024 * 4
        ) as error_handle:
            for fasta_id, sequence, error in tqdm(
                iterator,
                total=len(cif_paths),
                desc="Extracting receptor sequences",
                unit="file",
                smoothing=0.05,
            ):
                ok, fail = write_result(
                    fasta_handle=fasta_handle,
                    error_handle=error_handle,
                    fasta_id=fasta_id,
                    sequence=sequence,
                    error=error,
                )
                extracted_count += ok
                failed_count += fail

    print(f"Processed files: {len(cif_paths)}")
    print(f"FASTA entries written: {extracted_count}")
    print(f"Failed files: {failed_count}")
    print(f"Output FASTA: {output_path}")
    print(f"Error log: {error_path}")
    print(f"Model filter: {'disabled (--all-models)' if model_id is None else model_id}")


if __name__ == "__main__":
    main()
