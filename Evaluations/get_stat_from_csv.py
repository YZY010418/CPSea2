#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import sys
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a CSV/TSV file, compute mean and standard deviation of one column."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV/TSV file path"
    )
    parser.add_argument(
        "-c", "--column",
        required=True,
        type=int,
        help="1-based column index to read"
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Indicate the file has a header row (default: no header)"
    )
    parser.add_argument(
        "--type",
        choices=["csv", "tsv"],
        default="csv",
        help="Input file type: csv or tsv (default: csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output txt file; results will still be printed to stdout"
    )
    return parser.parse_args()


def count_lines(filepath: str, chunk_size: int = 1024 * 1024 * 8) -> int:
    """
    Fast line counting in binary mode.
    """
    total = 0
    with open(filepath, "rb") as f:
        with tqdm(
            desc="Counting lines",
            unit="B",
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total += chunk.count(b"\n")
                pbar.update(len(chunk))

            # Handle files not ending with '\n'
            f.seek(0, 2)
            file_size = f.tell()
            if file_size > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    total += 1
    return total


def compute_stats(filepath: str, column_index_1based: int, has_header: bool, file_type: str):
    """
    Stream through the file and compute mean/std using Welford's algorithm.
    Returns a dict of summary stats.
    """
    delimiter = "," if file_type == "csv" else "\t"
    col_idx = column_index_1based - 1

    total_lines = count_lines(filepath)
    total_rows_to_process = total_lines - 1 if has_header and total_lines > 0 else total_lines

    n = 0
    mean = 0.0
    m2 = 0.0

    skipped_empty = 0
    skipped_short = 0
    skipped_non_numeric = 0

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            try:
                next(reader)
            except StopIteration:
                raise ValueError("Input file is empty.")

        with tqdm(total=total_rows_to_process, desc="Processing rows", unit="row") as pbar:
            for row_num, row in enumerate(reader, start=(2 if has_header else 1)):
                pbar.update(1)

                if not row:
                    skipped_empty += 1
                    continue

                if col_idx >= len(row):
                    skipped_short += 1
                    continue

                raw = row[col_idx].strip()
                if raw == "":
                    skipped_empty += 1
                    continue

                try:
                    x = float(raw)
                except ValueError:
                    skipped_non_numeric += 1
                    continue

                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2

    if n == 0:
        raise ValueError("No valid numeric values found in the specified column.")

    # Sample std (ddof=1)
    std_sample = math.sqrt(m2 / (n - 1)) if n > 1 else float("nan")
    # Population std (ddof=0)
    std_population = math.sqrt(m2 / n)

    return {
        "input_file": filepath,
        "column_1based": column_index_1based,
        "file_type": file_type,
        "has_header": has_header,
        "total_lines": total_lines,
        "rows_examined": total_rows_to_process,
        "valid_numeric_rows": n,
        "mean": mean,
        "std_sample_ddof1": std_sample,
        "std_population_ddof0": std_population,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_non_numeric": skipped_non_numeric,
    }


def format_result(stats: dict) -> str:
    lines = [
        f"input_file           : {stats['input_file']}",
        f"column_1based        : {stats['column_1based']}",
        f"file_type            : {stats['file_type']}",
        f"has_header           : {stats['has_header']}",
        f"total_lines          : {stats['total_lines']}",
        f"rows_examined        : {stats['rows_examined']}",
        f"valid_numeric_rows   : {stats['valid_numeric_rows']}",
        f"mean                 : {stats['mean']:.10g}",
        f"std_sample_ddof1     : {stats['std_sample_ddof1']:.10g}",
        f"std_population_ddof0 : {stats['std_population_ddof0']:.10g}",
        f"skipped_empty        : {stats['skipped_empty']}",
        f"skipped_short        : {stats['skipped_short']}",
        f"skipped_non_numeric  : {stats['skipped_non_numeric']}",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.column < 1:
        print("Error: -c/--column must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        stats = compute_stats(
            filepath=args.input,
            column_index_1based=args.column,
            has_header=args.head,
            file_type=args.type,
        )
        result_text = format_result(stats)

        print(result_text)

        if args.output is not None:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(result_text + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
