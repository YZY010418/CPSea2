#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a CSV/TSV file, compute mean and standard deviation of one column. "
            "Optionally deduplicate CPSea2-style IDs by randomly keeping one row per "
            "'XXXX_clusterY' group. Optionally filter rows by peptide length range, "
            "either from the input file itself or from an external length file matched by ID. "
            "If --negative is specified, only values < 0 are used for statistics."
        )
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
        help="1-based column index to read as numeric values"
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Indicate the input file has a header row (default: no header)"
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
    parser.add_argument(
        "--cpsea2_id",
        type=int,
        default=None,
        help=(
            "Optional 1-based column index containing strings like "
            "'2QPW_cluster0_ligand2'. For rows sharing the same prefix "
            "'2QPW_cluster0' (first two underscore-separated fields), "
            "randomly keep only one row for statistics."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used together with --cpsea2_id (default: 0)"
    )
    parser.add_argument(
        "--length_column",
        type=int,
        default=None,
        help=(
            "Optional 1-based column index containing peptide length. "
            "If --length_file is not given, this column is read from the input file itself. "
            "If --length_file is given, this column is read from the external length file."
        )
    )
    parser.add_argument(
        "--min",
        dest="min_length",
        type=float,
        default=None,
        help="Optional inclusive minimum peptide length"
    )
    parser.add_argument(
        "--max",
        dest="max_length",
        type=float,
        default=None,
        help="Optional inclusive maximum peptide length"
    )
    parser.add_argument(
        "--length_file",
        default=None,
        help="Optional external CSV/TSV file used for length filtering by ID matching"
    )
    parser.add_argument(
        "--length_type",
        choices=["csv", "tsv"],
        default="csv",
        help="File type of --length_file: csv or tsv (default: csv)"
    )
    parser.add_argument(
        "--length_ID",
        type=int,
        default=None,
        help="1-based ID column index in --length_file"
    )
    parser.add_argument(
        "--length_head",
        action="store_true",
        help="Indicate --length_file has a header row (default: no header)"
    )
    parser.add_argument(
        "--ID",
        type=int,
        default=None,
        help="1-based ID column index in the input file, used together with --length_file"
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="If specified, only values < 0 are used for statistics, and their count is reported."
    )
    return parser.parse_args()


def count_lines(filepath: str, chunk_size: int = 1024 * 1024 * 8) -> int:
    total = 0
    with open(filepath, "rb") as f:
        with tqdm(
            desc=f"Counting lines: {Path(filepath).name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total += chunk.count(b"\n")
                pbar.update(len(chunk))

            f.seek(0, 2)
            file_size = f.tell()
            if file_size > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    total += 1
    return total


def parse_cpsea2_group(raw_value: str):
    if raw_value is None:
        return None

    raw_value = raw_value.strip()
    if raw_value == "":
        return None

    parts = raw_value.split("_")
    if len(parts) < 2:
        return None
    if parts[0] == "" or parts[1] == "":
        return None

    return f"{parts[0]}_{parts[1]}"


def normalize_match_id(raw_value: str):
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if value == "":
        return None

    suffixes = ["_relaxed_relaxed", "_relaxed", ".pdb"]
    changed = True
    while changed and value:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix):
                value = value[: -len(suffix)].rstrip()
                changed = True
                break

    return value if value else None


def check_length_value(length_value, min_length, max_length):
    if length_value is None:
        return False, "empty"

    try:
        length_value = float(length_value)
    except (TypeError, ValueError):
        return False, "non_numeric"

    if min_length is not None and length_value < min_length:
        return False, "out_of_range"

    if max_length is not None and length_value > max_length:
        return False, "out_of_range"

    return True, None


def check_length_filter(row, length_col_idx, min_length, max_length):
    if length_col_idx is None:
        return True, "no_length_filter"

    if not row:
        return False, "empty"

    if length_col_idx >= len(row):
        return False, "short"

    raw = row[length_col_idx].strip()
    if raw == "":
        return False, "empty"

    return check_length_value(raw, min_length, max_length)


def build_length_lookup(
    filepath: str,
    file_type: str,
    has_header: bool,
    id_col_1based: int,
    length_col_1based: int,
    total_lines: int,
):
    delimiter = "," if file_type == "csv" else "\t"
    id_col_idx = id_col_1based - 1
    length_col_idx = length_col_1based - 1
    total_data_rows = total_lines - 1 if has_header and total_lines > 0 else total_lines

    lookup = {}
    duplicate_ids = 0
    skipped_empty_id = 0
    skipped_short = 0
    skipped_empty_length = 0

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            try:
                next(reader)
            except StopIteration:
                raise ValueError("Length file is empty.")

        with tqdm(total=total_data_rows, desc="Loading length file", unit="row") as pbar:
            for row in reader:
                pbar.update(1)

                if not row:
                    skipped_empty_id += 1
                    continue

                if id_col_idx >= len(row) or length_col_idx >= len(row):
                    skipped_short += 1
                    continue

                raw_id = row[id_col_idx].strip()
                norm_id = normalize_match_id(raw_id)
                if norm_id is None:
                    skipped_empty_id += 1
                    continue

                raw_length = row[length_col_idx].strip()
                if raw_length == "":
                    skipped_empty_length += 1
                    continue

                if norm_id in lookup:
                    duplicate_ids += 1
                    continue

                lookup[norm_id] = raw_length

    return {
        "lookup": lookup,
        "length_file": filepath,
        "length_file_type": file_type,
        "length_file_has_header": has_header,
        "length_file_total_lines": total_lines,
        "length_file_data_rows_total": total_data_rows,
        "length_file_id_1based": id_col_1based,
        "length_file_length_column_1based": length_col_1based,
        "length_file_unique_ids": len(lookup),
        "length_file_duplicate_ids": duplicate_ids,
        "length_file_skipped_empty_id": skipped_empty_id,
        "length_file_skipped_short": skipped_short,
        "length_file_skipped_empty_length": skipped_empty_length,
    }


def apply_external_length_filter(
    row,
    input_id_col_idx,
    length_lookup,
    min_length,
    max_length,
):
    if not row:
        return False, "empty_input_id"

    if input_id_col_idx >= len(row):
        return False, "input_short"

    raw_input_id = row[input_id_col_idx].strip()
    norm_input_id = normalize_match_id(raw_input_id)
    if norm_input_id is None:
        return False, "empty_input_id"

    if norm_input_id not in length_lookup:
        return False, "unmatched"

    raw_length = length_lookup[norm_input_id]
    if raw_length is None or str(raw_length).strip() == "":
        return False, "matched_empty_length"

    passes, reason = check_length_value(raw_length, min_length, max_length)
    if passes:
        return True, None

    if reason == "empty":
        return False, "matched_empty_length"
    if reason == "non_numeric":
        return False, "matched_non_numeric_length"
    if reason == "out_of_range":
        return False, "matched_out_of_range"
    return False, reason


def select_cpsea2_rows(
    filepath: str,
    cpsea2_id_1based: int,
    has_header: bool,
    file_type: str,
    total_data_rows: int,
    seed: int,
    length_column_1based=None,
    min_length=None,
    max_length=None,
    input_id_1based=None,
    external_length_lookup=None,
):
    delimiter = "," if file_type == "csv" else "\t"
    cpsea2_col_idx = cpsea2_id_1based - 1
    length_col_idx = length_column_1based - 1 if (length_column_1based is not None and external_length_lookup is None) else None
    input_id_col_idx = input_id_1based - 1 if input_id_1based is not None else None
    rng = random.Random(seed)

    group_seen_count = {}
    group_selected_row = {}

    cpsea2_skipped_empty = 0
    cpsea2_skipped_short = 0
    cpsea2_skipped_invalid = 0

    length_filter_rows_passed = 0
    length_filter_skipped_empty = 0
    length_filter_skipped_short = 0
    length_filter_skipped_non_numeric = 0
    length_filter_skipped_out_of_range = 0
    length_filter_skipped_unmatched = 0

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            try:
                next(reader)
            except StopIteration:
                raise ValueError("Input file is empty.")

        with tqdm(total=total_data_rows, desc="Selecting CPSea2 rows", unit="row") as pbar:
            for data_row_idx, row in enumerate(reader, start=1):
                pbar.update(1)

                if external_length_lookup is not None:
                    passed_length, length_reason = apply_external_length_filter(
                        row=row,
                        input_id_col_idx=input_id_col_idx,
                        length_lookup=external_length_lookup,
                        min_length=min_length,
                        max_length=max_length,
                    )
                else:
                    passed_length, length_reason = check_length_filter(
                        row=row,
                        length_col_idx=length_col_idx,
                        min_length=min_length,
                        max_length=max_length,
                    )

                if not passed_length:
                    if length_reason in ("empty", "matched_empty_length", "empty_input_id"):
                        length_filter_skipped_empty += 1
                    elif length_reason in ("short", "input_short"):
                        length_filter_skipped_short += 1
                    elif length_reason == "non_numeric" or length_reason == "matched_non_numeric_length":
                        length_filter_skipped_non_numeric += 1
                    elif length_reason == "out_of_range" or length_reason == "matched_out_of_range":
                        length_filter_skipped_out_of_range += 1
                    elif length_reason == "unmatched":
                        length_filter_skipped_unmatched += 1
                    continue

                if length_col_idx is not None or external_length_lookup is not None:
                    length_filter_rows_passed += 1

                if not row:
                    cpsea2_skipped_empty += 1
                    continue

                if cpsea2_col_idx >= len(row):
                    cpsea2_skipped_short += 1
                    continue

                raw_id = row[cpsea2_col_idx].strip()
                if raw_id == "":
                    cpsea2_skipped_empty += 1
                    continue

                group_key = parse_cpsea2_group(raw_id)
                if group_key is None:
                    cpsea2_skipped_invalid += 1
                    continue

                seen = group_seen_count.get(group_key, 0) + 1
                group_seen_count[group_key] = seen

                if seen == 1 or rng.randrange(seen) == 0:
                    group_selected_row[group_key] = data_row_idx

    selected_row_indices = set(group_selected_row.values())

    return {
        "selected_row_indices": selected_row_indices,
        "cpsea2_unique_groups": len(group_selected_row),
        "cpsea2_rows_selected": len(selected_row_indices),
        "cpsea2_skipped_empty": cpsea2_skipped_empty,
        "cpsea2_skipped_short": cpsea2_skipped_short,
        "cpsea2_skipped_invalid": cpsea2_skipped_invalid,
        "length_filter_rows_passed": length_filter_rows_passed,
        "length_filter_skipped_empty": length_filter_skipped_empty,
        "length_filter_skipped_short": length_filter_skipped_short,
        "length_filter_skipped_non_numeric": length_filter_skipped_non_numeric,
        "length_filter_skipped_out_of_range": length_filter_skipped_out_of_range,
        "length_filter_skipped_unmatched": length_filter_skipped_unmatched,
    }


def compute_stats(
    filepath: str,
    column_index_1based: int,
    has_header: bool,
    file_type: str,
    total_lines: int,
    selected_row_indices=None,
    cpsea2_info=None,
    length_column_1based=None,
    min_length=None,
    max_length=None,
    input_id_1based=None,
    external_length_lookup=None,
    negative_only: bool = False,
):
    delimiter = "," if file_type == "csv" else "\t"
    col_idx = column_index_1based - 1
    length_col_idx = length_column_1based - 1 if (length_column_1based is not None and external_length_lookup is None) else None
    input_id_col_idx = input_id_1based - 1 if input_id_1based is not None else None
    external_length_lookup_map = (
        external_length_lookup["lookup"]
        if isinstance(external_length_lookup, dict) and "lookup" in external_length_lookup
        else external_length_lookup
    )

    total_data_rows = total_lines - 1 if has_header and total_lines > 0 else total_lines

    n = 0
    mean = 0.0
    m2 = 0.0

    skipped_empty = 0
    skipped_short = 0
    skipped_non_numeric = 0
    skipped_non_negative = 0

    rows_considered_for_stats = 0
    negative_value_rows = 0

    length_filter_rows_passed = 0
    length_filter_skipped_empty = 0
    length_filter_skipped_short = 0
    length_filter_skipped_non_numeric = 0
    length_filter_skipped_out_of_range = 0
    length_filter_skipped_unmatched = 0

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            try:
                next(reader)
            except StopIteration:
                raise ValueError("Input file is empty.")

        with tqdm(total=total_data_rows, desc="Processing rows", unit="row") as pbar:
            for data_row_idx, row in enumerate(reader, start=1):
                pbar.update(1)

                if selected_row_indices is not None and data_row_idx not in selected_row_indices:
                    continue

                if selected_row_indices is None:
                    if external_length_lookup_map is not None:
                        passed_length, length_reason = apply_external_length_filter(
                            row=row,
                            input_id_col_idx=input_id_col_idx,
                            length_lookup=external_length_lookup_map,
                            min_length=min_length,
                            max_length=max_length,
                        )
                    else:
                        passed_length, length_reason = check_length_filter(
                            row=row,
                            length_col_idx=length_col_idx,
                            min_length=min_length,
                            max_length=max_length,
                        )

                    if not passed_length:
                        if length_reason in ("empty", "matched_empty_length", "empty_input_id"):
                            length_filter_skipped_empty += 1
                        elif length_reason in ("short", "input_short"):
                            length_filter_skipped_short += 1
                        elif length_reason == "non_numeric" or length_reason == "matched_non_numeric_length":
                            length_filter_skipped_non_numeric += 1
                        elif length_reason == "out_of_range" or length_reason == "matched_out_of_range":
                            length_filter_skipped_out_of_range += 1
                        elif length_reason == "unmatched":
                            length_filter_skipped_unmatched += 1
                        continue

                    if length_col_idx is not None or external_length_lookup_map is not None:
                        length_filter_rows_passed += 1

                rows_considered_for_stats += 1

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

                if x < 0:
                    negative_value_rows += 1

                if negative_only and x >= 0:
                    skipped_non_negative += 1
                    continue

                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2

    if rows_considered_for_stats == 0:
        raise ValueError("No rows were selected for statistics.")

    if n == 0:
        if negative_only:
            raise ValueError("No valid numeric values < 0 were found in the specified column.")
        raise ValueError("No valid numeric values found in the specified column.")

    std_sample = math.sqrt(m2 / (n - 1)) if n > 1 else float("nan")
    std_population = math.sqrt(m2 / n)

    stats = {
        "input_file": filepath,
        "column_1based": column_index_1based,
        "file_type": file_type,
        "has_header": has_header,
        "total_lines": total_lines,
        "data_rows_total": total_data_rows,
        "rows_considered_for_stats": rows_considered_for_stats,
        "valid_numeric_rows": n,
        "mean": mean,
        "std_sample_ddof1": std_sample,
        "std_population_ddof0": std_population,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_non_numeric": skipped_non_numeric,
        "negative_only": negative_only,
        "negative_value_rows": negative_value_rows,
        "skipped_non_negative": skipped_non_negative,
        "cpsea2_id_1based": None,
        "cpsea2_unique_groups": None,
        "cpsea2_rows_selected": None,
        "cpsea2_skipped_empty": None,
        "cpsea2_skipped_short": None,
        "cpsea2_skipped_invalid": None,
        "cpsea2_seed": None,
        "length_column_1based": length_column_1based,
        "min_length": min_length,
        "max_length": max_length,
        "length_file": None,
        "length_file_type": None,
        "length_file_has_header": None,
        "length_file_id_1based": None,
        "input_id_1based": input_id_1based,
        "length_file_unique_ids": None,
        "length_file_duplicate_ids": None,
        "length_file_skipped_empty_id": None,
        "length_file_skipped_short": None,
        "length_file_skipped_empty_length": None,
        "length_filter_rows_passed": None,
        "length_filter_skipped_empty": None,
        "length_filter_skipped_short": None,
        "length_filter_skipped_non_numeric": None,
        "length_filter_skipped_out_of_range": None,
        "length_filter_skipped_unmatched": None,
    }

    if cpsea2_info is not None:
        stats.update({
            "cpsea2_id_1based": cpsea2_info["cpsea2_id_1based"],
            "cpsea2_unique_groups": cpsea2_info["cpsea2_unique_groups"],
            "cpsea2_rows_selected": cpsea2_info["cpsea2_rows_selected"],
            "cpsea2_skipped_empty": cpsea2_info["cpsea2_skipped_empty"],
            "cpsea2_skipped_short": cpsea2_info["cpsea2_skipped_short"],
            "cpsea2_skipped_invalid": cpsea2_info["cpsea2_skipped_invalid"],
            "cpsea2_seed": cpsea2_info["cpsea2_seed"],
            "length_filter_rows_passed": cpsea2_info["length_filter_rows_passed"],
            "length_filter_skipped_empty": cpsea2_info["length_filter_skipped_empty"],
            "length_filter_skipped_short": cpsea2_info["length_filter_skipped_short"],
            "length_filter_skipped_non_numeric": cpsea2_info["length_filter_skipped_non_numeric"],
            "length_filter_skipped_out_of_range": cpsea2_info["length_filter_skipped_out_of_range"],
            "length_filter_skipped_unmatched": cpsea2_info["length_filter_skipped_unmatched"],
        })
    elif length_column_1based is not None or external_length_lookup_map is not None:
        stats.update({
            "length_filter_rows_passed": length_filter_rows_passed,
            "length_filter_skipped_empty": length_filter_skipped_empty,
            "length_filter_skipped_short": length_filter_skipped_short,
            "length_filter_skipped_non_numeric": length_filter_skipped_non_numeric,
            "length_filter_skipped_out_of_range": length_filter_skipped_out_of_range,
            "length_filter_skipped_unmatched": length_filter_skipped_unmatched,
        })

    if external_length_lookup is not None:
        stats.update({
            "length_file": external_length_lookup["length_file"],
            "length_file_type": external_length_lookup["length_file_type"],
            "length_file_has_header": external_length_lookup["length_file_has_header"],
            "length_file_id_1based": external_length_lookup["length_file_id_1based"],
            "length_file_unique_ids": external_length_lookup["length_file_unique_ids"],
            "length_file_duplicate_ids": external_length_lookup["length_file_duplicate_ids"],
            "length_file_skipped_empty_id": external_length_lookup["length_file_skipped_empty_id"],
            "length_file_skipped_short": external_length_lookup["length_file_skipped_short"],
            "length_file_skipped_empty_length": external_length_lookup["length_file_skipped_empty_length"],
        })

    return stats


def format_result(stats: dict) -> str:
    lines = [
        f"input_file               : {stats['input_file']}",
        f"column_1based            : {stats['column_1based']}",
        f"file_type                : {stats['file_type']}",
        f"has_header               : {stats['has_header']}",
        f"total_lines              : {stats['total_lines']}",
        f"data_rows_total          : {stats['data_rows_total']}",
        f"rows_considered_for_stats: {stats['rows_considered_for_stats']}",
        f"valid_numeric_rows       : {stats['valid_numeric_rows']}",
        f"mean                     : {stats['mean']:.10g}",
        f"std_sample_ddof1         : {stats['std_sample_ddof1']:.10g}",
        f"std_population_ddof0     : {stats['std_population_ddof0']:.10g}",
        f"skipped_empty            : {stats['skipped_empty']}",
        f"skipped_short            : {stats['skipped_short']}",
        f"skipped_non_numeric      : {stats['skipped_non_numeric']}",
    ]

    if stats["negative_only"]:
        lines.extend([
            f"negative_only            : {stats['negative_only']}",
            f"negative_value_rows      : {stats['negative_value_rows']}",
            f"skipped_non_negative     : {stats['skipped_non_negative']}",
        ])

    if stats["length_column_1based"] is not None:
        lines.extend([
            f"length_column_1based     : {stats['length_column_1based']}",
            f"min_length               : {stats['min_length']}",
            f"max_length               : {stats['max_length']}",
            f"length_filter_rows_passed: {stats['length_filter_rows_passed']}",
            f"length_skipped_empty     : {stats['length_filter_skipped_empty']}",
            f"length_skipped_short     : {stats['length_filter_skipped_short']}",
            f"length_skipped_non_numeric: {stats['length_filter_skipped_non_numeric']}",
            f"length_skipped_out_of_range: {stats['length_filter_skipped_out_of_range']}",
            f"length_skipped_unmatched : {stats['length_filter_skipped_unmatched']}",
        ])

    if stats["length_file"] is not None:
        lines.extend([
            f"length_file              : {stats['length_file']}",
            f"length_file_type         : {stats['length_file_type']}",
            f"length_file_has_header   : {stats['length_file_has_header']}",
            f"input_id_1based          : {stats['input_id_1based']}",
            f"length_file_id_1based    : {stats['length_file_id_1based']}",
            f"length_file_unique_ids   : {stats['length_file_unique_ids']}",
            f"length_file_duplicate_ids: {stats['length_file_duplicate_ids']}",
            f"length_file_skipped_empty_id: {stats['length_file_skipped_empty_id']}",
            f"length_file_skipped_short: {stats['length_file_skipped_short']}",
            f"length_file_skipped_empty_length: {stats['length_file_skipped_empty_length']}",
        ])

    if stats["cpsea2_id_1based"] is not None:
        lines.extend([
            f"cpsea2_id_1based         : {stats['cpsea2_id_1based']}",
            f"cpsea2_unique_groups     : {stats['cpsea2_unique_groups']}",
            f"cpsea2_rows_selected     : {stats['cpsea2_rows_selected']}",
            f"cpsea2_skipped_empty     : {stats['cpsea2_skipped_empty']}",
            f"cpsea2_skipped_short     : {stats['cpsea2_skipped_short']}",
            f"cpsea2_skipped_invalid   : {stats['cpsea2_skipped_invalid']}",
            f"cpsea2_seed              : {stats['cpsea2_seed']}",
        ])

    return "\n".join(lines)



def main():
    args = parse_args()

    if args.column < 1:
        print("Error: -c/--column must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.cpsea2_id is not None and args.cpsea2_id < 1:
        print("Error: --cpsea2_id must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.length_column is not None and args.length_column < 1:
        print("Error: --length_column must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.ID is not None and args.ID < 1:
        print("Error: --ID must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.length_ID is not None and args.length_ID < 1:
        print("Error: --length_ID must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.min_length is not None and args.max_length is not None:
        if args.min_length > args.max_length:
            print("Error: --min cannot be greater than --max", file=sys.stderr)
            sys.exit(1)

    if args.length_file is not None:
        if args.ID is None:
            print("Error: --length_file requires --ID", file=sys.stderr)
            sys.exit(1)
        if args.length_ID is None:
            print("Error: --length_file requires --length_ID", file=sys.stderr)
            sys.exit(1)
        if args.length_column is None:
            print("Error: --length_file requires --length_column", file=sys.stderr)
            sys.exit(1)
    else:
        if args.length_ID is not None:
            print("Error: --length_ID requires --length_file", file=sys.stderr)
            sys.exit(1)
        if args.length_head:
            print("Error: --length_head requires --length_file", file=sys.stderr)
            sys.exit(1)
        if args.ID is not None:
            print("Error: --ID requires --length_file", file=sys.stderr)
            sys.exit(1)
        if (args.min_length is not None or args.max_length is not None) and args.length_column is None:
            print("Error: --min/--max require --length_column", file=sys.stderr)
            sys.exit(1)

    try:
        total_lines = count_lines(args.input)
        total_data_rows = total_lines - 1 if args.head and total_lines > 0 else total_lines

        external_length_info = None
        if args.length_file is not None:
            length_total_lines = count_lines(args.length_file)
            external_length_info = build_length_lookup(
                filepath=args.length_file,
                file_type=args.length_type,
                has_header=args.length_head,
                id_col_1based=args.length_ID,
                length_col_1based=args.length_column,
                total_lines=length_total_lines,
            )

        selected_row_indices = None
        cpsea2_info = None

        if args.cpsea2_id is not None:
            cpsea2_select_result = select_cpsea2_rows(
                filepath=args.input,
                cpsea2_id_1based=args.cpsea2_id,
                has_header=args.head,
                file_type=args.type,
                total_data_rows=total_data_rows,
                seed=args.seed,
                length_column_1based=args.length_column,
                min_length=args.min_length,
                max_length=args.max_length,
                input_id_1based=args.ID,
                external_length_lookup=(external_length_info["lookup"] if external_length_info is not None else None),
            )
            selected_row_indices = cpsea2_select_result["selected_row_indices"]

            cpsea2_info = {
                "cpsea2_id_1based": args.cpsea2_id,
                "cpsea2_unique_groups": cpsea2_select_result["cpsea2_unique_groups"],
                "cpsea2_rows_selected": cpsea2_select_result["cpsea2_rows_selected"],
                "cpsea2_skipped_empty": cpsea2_select_result["cpsea2_skipped_empty"],
                "cpsea2_skipped_short": cpsea2_select_result["cpsea2_skipped_short"],
                "cpsea2_skipped_invalid": cpsea2_select_result["cpsea2_skipped_invalid"],
                "cpsea2_seed": args.seed,
                "length_filter_rows_passed": cpsea2_select_result["length_filter_rows_passed"],
                "length_filter_skipped_empty": cpsea2_select_result["length_filter_skipped_empty"],
                "length_filter_skipped_short": cpsea2_select_result["length_filter_skipped_short"],
                "length_filter_skipped_non_numeric": cpsea2_select_result["length_filter_skipped_non_numeric"],
                "length_filter_skipped_out_of_range": cpsea2_select_result["length_filter_skipped_out_of_range"],
                "length_filter_skipped_unmatched": cpsea2_select_result["length_filter_skipped_unmatched"],
            }

        stats = compute_stats(
            filepath=args.input,
            column_index_1based=args.column,
            has_header=args.head,
            file_type=args.type,
            total_lines=total_lines,
            selected_row_indices=selected_row_indices,
            cpsea2_info=cpsea2_info,
            length_column_1based=args.length_column,
            min_length=args.min_length,
            max_length=args.max_length,
            input_id_1based=args.ID,
            external_length_lookup=external_length_info,
            negative_only=args.negative,
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
