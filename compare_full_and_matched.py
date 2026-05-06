#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_tsv(path, no_header=False):
    if no_header:
        return pd.read_csv(path, sep="\t", header=None)
    return pd.read_csv(path, sep="\t")


def to_numeric_array(df, col_idx):
    x = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").dropna().to_numpy()
    return x[np.isfinite(x)]


def relative_hist(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(values, bins=bins)
    return counts / counts.sum()


def main():
    parser = argparse.ArgumentParser(
        description="Compare frequency distributions of columns 3-12 from two TSV files."
    )
    parser.add_argument("--tsv1", required=True, help="First input TSV file, e.g. all CPbase")
    parser.add_argument("--tsv2", required=True, help="Second input TSV file, e.g. matched CPbase")
    parser.add_argument("-o", "--out", required=True, help="Output figure path, e.g. dist_compare.png")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--no-header", action="store_true", help="Use this if TSV files do not contain header")
    parser.add_argument("--label1", default="All CPbase", help="Label for the first TSV")
    parser.add_argument("--label2", default="Matched", help="Label for the second TSV")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI")
    args = parser.parse_args()

    df1 = read_tsv(args.tsv1, args.no_header)
    df2 = read_tsv(args.tsv2, args.no_header)

    if df1.shape[1] < 12 or df2.shape[1] < 12:
        raise ValueError("Both TSV files must contain at least 12 columns.")

    # zero-based indices for 1-based columns 3-12
    col_indices = list(range(2, 12))

    # Corresponding variables:
    # f"{length}\t"
    # f"{abs_bsa:.2f}\t"
    # f"{rel_bsa:.4f}\t"
    # f"{rel_apolar_bsa:.4f}\t"
    # f"{bsa_per_length:.4f}\t"
    # f"{rel_n_rBSA:.4f}\t"
    # f"{rel_c_rBSA:.4f}\t"
    # f"{gravy:.4f}\t"
    # f"{helix_ratio:.4f}\t"
    # f"{beta_ratio:.4f}\t"
    metric_names = [
        "length",
        "abs_bsa",
        "rel_bsa",
        "rel_apolar_bsa",
        "bsa_per_length",
        "rel_n_rBSA",
        "rel_c_rBSA",
        "gravy",
        "helix_ratio",
        "beta_ratio",
    ]

    color1 = "#8aabd4"  # All CPbase
    color2 = "#b97ab1"  # Matched

    fig, axes = plt.subplots(
        4, 5,
        figsize=(18, 11),
        sharex=False,
        sharey=False,
        constrained_layout=True
    )

    for j, col_idx in enumerate(col_indices):
        x1 = to_numeric_array(df1, col_idx)
        x2 = to_numeric_array(df2, col_idx)

        combined = np.concatenate([x1, x2])
        if len(combined) == 0:
            print(f"Warning: column {col_idx + 1} contains no valid numeric values in either file.")
            continue

        vmin, vmax = np.min(combined), np.max(combined)

        if vmin == vmax:
            delta = 0.5 if vmin == 0 else abs(vmin) * 0.05
            vmin -= delta
            vmax += delta

        bins = np.linspace(vmin, vmax, args.bins + 1)

        h1 = relative_hist(x1, bins)
        h2 = relative_hist(x2, bins)

        ymax = max(h1.max(), h2.max())
        if ymax == 0:
            ymax = 1.0
        else:
            ymax *= 1.10

        bin_width = bins[1] - bins[0]

        # First five metrics: rows 0-1
        # Last five metrics: rows 2-3
        panel_group = j // 5
        panel_col = j % 5
        row1 = panel_group * 2
        row2 = row1 + 1

        ax1 = axes[row1, panel_col]
        ax2 = axes[row2, panel_col]

        ax1.bar(bins[:-1], h1, width=bin_width, align="edge", alpha=0.85, color=color1)
        ax2.bar(bins[:-1], h2, width=bin_width, align="edge", alpha=0.85, color=color2)

        ax1.set_xlim(vmin, vmax)
        ax2.set_xlim(vmin, vmax)

        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

        ax1.set_title(metric_names[j], fontsize=11)

        if panel_col == 0:
            ax1.set_ylabel(args.label1)
            ax2.set_ylabel(args.label2)

        ax2.set_xlabel("Value")

        ax1.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax2.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        ax2.tick_params(axis="y", labelsize=8)

    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
