#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def read_rmsd_from_jsonl(jsonl_path):
    """Read rmsd values from a results.jsonl file."""
    values = []

    if not jsonl_path.exists():
        return values

    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Invalid JSON: {jsonl_path}, line {line_no}")
                continue

            if "rmsd" not in record:
                continue

            try:
                rmsd = float(record["rmsd"])
            except (TypeError, ValueError):
                continue

            if np.isfinite(rmsd):
                values.append(rmsd)

    return values


def collect_cyclization_types(input_dir, scales):
    """Collect all cyclization type folder names from scale folders."""
    cyclization_types = set()

    for scale in scales:
        scale_dir = input_dir / f"scale_{scale:.1f}"
        if not scale_dir.exists():
            continue

        for subdir in scale_dir.iterdir():
            if subdir.is_dir():
                cyclization_types.add(subdir.name)

    return sorted(cyclization_types)


def collect_data(input_dir, scales, cyclization_types):
    """
    Return:
        data[cyclization_type][scale] = list of rmsd values
    """
    data = {}

    for ctype in cyclization_types:
        data[ctype] = {}

        for scale in scales:
            jsonl_path = (
                input_dir
                / f"scale_{scale:.1f}"
                / ctype
                / "candidates"
                / "results.jsonl"
            )

            rmsd_values = read_rmsd_from_jsonl(jsonl_path)
            data[ctype][scale] = rmsd_values

            if len(rmsd_values) == 0:
                print(f"[Warning] No RMSD values found: {jsonl_path}")
            else:
                print(
                    f"[Info] {ctype:24s} scale={scale:.1f}: "
                    f"{len(rmsd_values)} RMSD values"
                )

    return data


def build_all_data(data, scales):
    """
    Merge RMSD values from all cyclization types for each scale.

    Return:
        all_data["ALL"][scale] = merged list of rmsd values
    """
    all_data = {"ALL": {}}

    for scale in scales:
        merged = []
        for ctype in data:
            merged.extend(data[ctype].get(scale, []))
        all_data["ALL"][scale] = merged

    return all_data


def print_rmsd_summary(data, scales, threshold=1.0):
    """
    Print:
      1. Per cyclization type and scale:
         - N
         - mean RMSD
         - N_pass
         - success ratio

      2. Per scale summary across all cyclization types:
         - mean of per-type success ratios
         - pooled success ratio after merging all samples
         - pooled RMSD mean
    """
    print()
    print(f"[Summary] RMSD statistics by cyclization type and scale")
    print(f"[Criterion] Success = RMSD < {threshold:.2f} Å")
    print("-" * 96)
    print(
        f"{'Cyclization type':28s} "
        f"{'Scale':>8s} "
        f"{'N':>8s} "
        f"{'RMSD mean':>12s} "
        f"{'N_pass':>8s} "
        f"{'Success':>12s}"
    )
    print("-" * 96)

    per_scale_type_ratios = {scale: [] for scale in scales}
    per_scale_pooled_values = {scale: [] for scale in scales}

    for ctype in data:
        for scale in scales:
            values = np.asarray(data[ctype].get(scale, []), dtype=float)
            n_total = len(values)

            if n_total == 0:
                print(
                    f"{ctype:28s} "
                    f"{scale:8.1f} "
                    f"{0:8d} "
                    f"{'NA':>12s} "
                    f"{0:8d} "
                    f"{'NA':>12s}"
                )
                continue

            mean_rmsd = float(np.mean(values))
            n_pass = int(np.sum(values < threshold))
            ratio = n_pass / n_total * 100.0

            per_scale_type_ratios[scale].append(ratio)
            per_scale_pooled_values[scale].extend(values.tolist())

            print(
                f"{ctype:28s} "
                f"{scale:8.1f} "
                f"{n_total:8d} "
                f"{mean_rmsd:12.4f} "
                f"{n_pass:8d} "
                f"{ratio:11.2f}%"
            )

    print("-" * 96)
    print()

    print("[Summary] Scale-level statistics across all cyclization types")
    print("-" * 108)
    print(
        f"{'Scale':>8s} "
        f"{'N_total':>10s} "
        f"{'Pooled RMSD mean':>18s} "
        f"{'Mean success of types':>24s} "
        f"{'Pooled success':>18s}"
    )
    print("-" * 108)

    for scale in scales:
        values = np.asarray(per_scale_pooled_values[scale], dtype=float)
        ratios = np.asarray(per_scale_type_ratios[scale], dtype=float)

        if len(values) == 0:
            print(
                f"{scale:8.1f} "
                f"{0:10d} "
                f"{'NA':>18s} "
                f"{'NA':>24s} "
                f"{'NA':>18s}"
            )
            continue

        pooled_mean = float(np.mean(values))
        pooled_success = float(np.mean(values < threshold) * 100.0)

        if len(ratios) > 0:
            mean_success_of_types = float(np.mean(ratios))
            mean_success_str = f"{mean_success_of_types:.2f}%"
        else:
            mean_success_str = "NA"

        print(
            f"{scale:8.1f} "
            f"{len(values):10d} "
            f"{pooled_mean:18.4f} "
            f"{mean_success_str:>24s} "
            f"{pooled_success:17.2f}%"
        )

    print("-" * 108)
    print()


def plot_data(
    data,
    scales,
    output_path,
    ncols=3,
    jitter=0.04,
    dpi=300,
    ylim=None,
    include_all=True,
):
    """
    Plot RMSD distributions.

    If include_all=True:
        - Adds one extra subplot named ALL.
        - Total subplots = number of cyclization types + 1.
    """
    plot_data_dict = dict(data)

    if include_all:
        all_data = build_all_data(data, scales)
        plot_data_dict.update(all_data)

    cyclization_types = list(plot_data_dict.keys())
    n_types = len(cyclization_types)

    if n_types == 0:
        raise ValueError("No cyclization type folders found.")

    nrows = int(np.ceil(n_types / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(123)

    # Pastel blue / purple / pink palette
    bar_colors = [
        "#B9DDF2",  # pastel blue
        "#C9B7EA",  # pastel purple
        "#F4C6D7",  # pastel pink
        "#A7D8E8",  # blue-cyan pastel
        "#D6C3F1",  # light purple
        "#EFB7CC",  # rose pink
    ]

    point_colors = [
        "#5E9FC7",
        "#8B6FC4",
        "#C7799B",
        "#4F9CAF",
        "#9A7CCF",
        "#C9658E",
    ]

    if len(scales) > 1:
        min_gap = np.min(np.diff(sorted(scales)))
        bar_width = 0.68 * min_gap
    else:
        bar_width = 0.35

    for ax_idx, ctype in enumerate(cyclization_types):
        ax = axes[ax_idx]

        for scale_idx, scale in enumerate(scales):
            values = np.asarray(plot_data_dict[ctype].get(scale, []), dtype=float)

            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))
            color_idx = scale_idx % len(bar_colors)

            # Mean RMSD as bar
            ax.bar(
                scale,
                mean_value,
                width=bar_width,
                color=bar_colors[color_idx],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.88,
                zorder=1,
            )

            # Raw RMSD values as jittered scatter
            x = np.full(len(values), scale, dtype=float)
            x_jittered = x + rng.normal(loc=0.0, scale=jitter, size=len(values))

            ax.scatter(
                x_jittered,
                values,
                s=13,
                color=point_colors[color_idx],
                alpha=0.50,
                linewidths=0,
                zorder=2,
            )

        if ctype == "ALL":
            ax.set_title("ALL cyclization types", fontweight="bold")
        else:
            ax.set_title(ctype)

        ax.set_xticks(scales)
        ax.set_xlabel("CFG scale")
        ax.set_ylabel("RMSD")

        # Remove grid lines
        ax.grid(False)

        # Cleaner visual style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

        if ylim is not None:
            ax.set_ylim(ylim)

    for j in range(n_types, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[Done] Figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot RMSD distributions across CFG scales for different cyclization types. "
            "The script also reports RMSD means and success ratios."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input folder containing scale_0.0, scale_0.5, ..., scale_2.5 folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="rmsd_by_scale.png",
        help="Output figure path. Default: rmsd_by_scale.png",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=DEFAULT_SCALES,
        help="Scale values. Default: 0.0 0.5 1.0 1.5 2.0 2.5",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help=(
            "Number of subplot columns. Default: 3. "
            "For 8 cyclization types + ALL, this gives a 3 x 3 layout."
        ),
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.04,
        help="Horizontal jitter for scatter points. Default: 0.04",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI. Default: 300",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits, e.g. --ylim 0 10",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=1.0,
        help="Threshold for reporting success ratio. Default: 1.0 Å",
    )
    parser.add_argument(
        "--no-all-panel",
        action="store_true",
        help="Disable the extra ALL cyclization types subplot.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    scales = sorted(args.scales)

    cyclization_types = collect_cyclization_types(input_dir, scales)
    print(f"[Info] Found {len(cyclization_types)} cyclization types:")
    for ctype in cyclization_types:
        print(f"  - {ctype}")

    data = collect_data(input_dir, scales, cyclization_types)

    print_rmsd_summary(
        data=data,
        scales=scales,
        threshold=args.rmsd_threshold,
    )

    plot_data(
        data=data,
        scales=scales,
        output_path=output_path,
        ncols=args.ncols,
        jitter=args.jitter,
        dpi=args.dpi,
        ylim=args.ylim,
        include_all=(not args.no_all_panel),
    )


if __name__ == "__main__":
    main()
