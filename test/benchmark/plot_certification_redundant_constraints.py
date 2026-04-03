#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "build/test/benchmark/certification_redundant_constraints_montecarlo.csv"
    )
    out_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "build/test/benchmark/certification_redundant_constraints"
    )

    df = pd.read_csv(csv_path)
    eps = 1e-16

    with_gap = np.log10(df["with_gap"].to_numpy() + eps)
    without_gap = np.log10(df["without_gap"].to_numpy() + eps)

    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.5),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2)

    bins_gap = np.linspace(
        min(with_gap.min(), without_gap.min()),
        max(without_gap.max(), with_gap.max()),
        16,
    )
    axes[0].hist(with_gap, bins=bins_gap, alpha=0.75, label="With redundant constraints", color="#1f77b4")
    axes[0].hist(without_gap, bins=bins_gap, alpha=0.60, label="Without redundant constraints", color="#d62728")
    axes[0].set_title("Certification Gap Distribution")
    axes[0].set_xlabel(r"$\log_{10}(\mathrm{best\ suboptimality} + 10^{-16})$")
    axes[0].set_ylabel("Trials")
    axes[0].legend(frameon=False)

    runtime_bins = np.linspace(0, max(df["with_runtime_ms"].max(), df["without_runtime_ms"].max()), 16)
    axes[1].hist(
        df["with_runtime_ms"],
        bins=runtime_bins,
        alpha=0.75,
        label="With redundant constraints",
        color="#1f77b4",
    )
    axes[1].hist(
        df["without_runtime_ms"],
        bins=runtime_bins,
        alpha=0.60,
        label="Without redundant constraints",
        color="#d62728",
    )
    axes[1].set_title("Runtime Distribution")
    axes[1].set_xlabel("Runtime (ms)")
    axes[1].set_ylabel("Trials")

    gap_ratio = np.median(df["without_gap"] / df["with_gap"])
    runtime_speedup = np.median(df["with_runtime_ms"] / df["without_runtime_ms"])
    with_rate = 100.0 * df["with_certified"].mean()
    without_rate = 100.0 * df["without_certified"].mean()
    axes[1].text(
        0.98,
        0.98,
        (
            f"Median gap inflation: {gap_ratio:.2f}x\n"
            f"Median runtime speedup: {runtime_speedup:.1f}x\n"
            f"Certified: {with_rate:.0f}% vs {without_rate:.0f}%"
        ),
        ha="right",
        va="top",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )

    fig.suptitle("Effect of Redundant SDP Constraints in the TEASER++ Certifier", y=1.02)
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
