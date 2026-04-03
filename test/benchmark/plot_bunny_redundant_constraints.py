#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_gap_plot(certifier_csv: Path, out_prefix: Path) -> None:
    df = pd.read_csv(certifier_csv)
    eps = 1e-16
    with_gap = np.log10(df["with_gap"].to_numpy() + eps)
    without_gap = np.log10(df["without_gap"].to_numpy() + eps)

    plt.rcParams.update(
        {
            "figure.figsize": (6.2, 4.2),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots()
    bins = np.linspace(min(with_gap.min(), without_gap.min()), max(with_gap.max(), without_gap.max()), 14)
    ax.hist(with_gap, bins=bins, alpha=0.75, color="#1f77b4", label="With redundant constraints")
    ax.hist(without_gap, bins=bins, alpha=0.60, color="#d62728", label="Without redundant constraints")
    ax.set_title("Bunny Suboptimality Gap Distribution")
    ax.set_xlabel(r"$\log_{10}(\mathrm{best\ suboptimality} + 10^{-16})$")
    ax.set_ylabel("Trials")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")


def save_rank_plot(rank_csv: Path, out_prefix: Path) -> None:
    df = pd.read_csv(rank_csv)
    ranks = sorted(set(df["with_rank"]).union(set(df["without_rank"])))
    with_counts = [(df["with_rank"] == r).sum() for r in ranks]
    without_counts = [(df["without_rank"] == r).sum() for r in ranks]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(ranks))
    width = 0.38
    ax.bar(x - width / 2, with_counts, width=width, color="#1f77b4", label="With redundant constraints")
    ax.bar(x + width / 2, without_counts, width=width, color="#d62728", label="Without redundant constraints")
    ax.set_title("Bunny SDP Rank Distribution")
    ax.set_xlabel("Numerical rank of SDP solution")
    ax.set_ylabel("Trials")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")


def main() -> int:
    certifier_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/test/benchmark/bunny_certifier_suboptimality.csv")
    rank_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("build/test/benchmark/bunny_sdp_rank_distribution.csv")
    gap_prefix = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("build/test/benchmark/bunny_suboptimality_gap_distribution")
    rank_prefix = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("build/test/benchmark/bunny_sdp_rank_distribution")

    gap_prefix.parent.mkdir(parents=True, exist_ok=True)
    rank_prefix.parent.mkdir(parents=True, exist_ok=True)

    save_gap_plot(certifier_csv, gap_prefix)
    save_rank_plot(rank_csv, rank_prefix)

    print(f"Wrote {gap_prefix.with_suffix('.png')}")
    print(f"Wrote {gap_prefix.with_suffix('.pdf')}")
    print(f"Wrote {rank_prefix.with_suffix('.png')}")
    print(f"Wrote {rank_prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
