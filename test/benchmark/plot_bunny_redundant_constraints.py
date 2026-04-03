#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_STATUSES = {"optimal", "optimal_inaccurate"}
COLORS = {"with": "#1f77b4", "without": "#d62728"}
LABELS = {"with": "With redundant constraints", "without": "Without redundant constraints"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.2, 4.2),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def successful_series(df: pd.DataFrame, prefix: str, column: str) -> np.ndarray:
    mask = df[f"{prefix}_status"].isin(VALID_STATUSES)
    return df.loc[mask, f"{prefix}_{column}"].to_numpy(dtype=float)


def continuous_bins(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
      return np.array([low - 0.5, high + 0.5])
    return np.linspace(low, high, min(16, max(8, len(values) // 4 + 1)))


def rank_ticks(rank_values: np.ndarray) -> np.ndarray:
    low = int(rank_values.min())
    high = int(rank_values.max())
    span = high - low + 1
    if span <= 12:
        step = 1
    elif span <= 24:
        step = 2
    else:
        step = 4
    ticks = np.arange(low, high + 1, step)
    if ticks[-1] != high:
        ticks = np.append(ticks, high)
    return ticks


def save_gap_hist(df: pd.DataFrame, prefix: str, out_path: Path) -> None:
    values = successful_series(df, prefix, "relative_gap")
    eps = 1e-16
    log_values = np.log10(np.maximum(values, eps))
    bins = continuous_bins(log_values)

    fig, ax = plt.subplots()
    ax.hist(log_values, bins=bins, color=COLORS[prefix], alpha=0.85)
    ax.set_title(f"Bunny SDP Relative Gap Histogram ({LABELS[prefix]})")
    ax.set_xlabel(r"$\log_{10}(\mathrm{relative\ cost\ gap} + 10^{-16})$")
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_gap_overlay_hist(df: pd.DataFrame, out_path: Path) -> None:
    eps = 1e-16
    with_values = successful_series(df, "with", "relative_gap")
    without_values = successful_series(df, "without", "relative_gap")
    with_log = np.log10(np.maximum(with_values, eps))
    without_log = np.log10(np.maximum(without_values, eps))
    bins = continuous_bins(np.concatenate([with_log, without_log]))

    fig, ax = plt.subplots()
    ax.hist(with_log, bins=bins, color=COLORS["with"], alpha=0.72, label=LABELS["with"])
    ax.hist(
        without_log,
        bins=bins,
        color=COLORS["without"],
        alpha=0.55,
        label=LABELS["without"],
    )
    ax.set_title("Bunny SDP Relative Gap Histogram")
    ax.set_xlabel(r"$\log_{10}(\mathrm{relative\ cost\ gap} + 10^{-16})$")
    ax.set_ylabel("Trials")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_runtime_hist(df: pd.DataFrame, prefix: str, out_path: Path) -> None:
    values = successful_series(df, prefix, "runtime_ms")
    bins = continuous_bins(values)

    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, color=COLORS[prefix], alpha=0.85)
    ax.set_title(f"Bunny SDP Runtime Histogram ({LABELS[prefix]})")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_runtime_overlay_hist(df: pd.DataFrame, out_path: Path) -> None:
    with_values = successful_series(df, "with", "runtime_ms")
    without_values = successful_series(df, "without", "runtime_ms")
    bins = continuous_bins(np.concatenate([with_values, without_values]))

    fig, ax = plt.subplots()
    ax.hist(with_values, bins=bins, color=COLORS["with"], alpha=0.72, label=LABELS["with"])
    ax.hist(
        without_values,
        bins=bins,
        color=COLORS["without"],
        alpha=0.55,
        label=LABELS["without"],
    )
    ax.set_title("Bunny SDP Runtime Histogram")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel("Trials")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_rank_hist(df: pd.DataFrame, prefix: str, out_path: Path) -> None:
    values = successful_series(df, prefix, "rank")
    rank_values = values.astype(int)
    bins = np.arange(rank_values.min() - 0.5, rank_values.max() + 1.5, 1.0)
    ticks = rank_ticks(rank_values)

    fig, ax = plt.subplots()
    ax.hist(rank_values, bins=bins, color=COLORS[prefix], alpha=0.9, rwidth=0.85)
    ax.set_title(f"Bunny SDP Rank Histogram ({LABELS[prefix]})")
    ax.set_xlabel("Numerical rank of SDP solution")
    ax.set_ylabel("Trials")
    ax.set_xticks(ticks)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_rank_overlay_hist(df: pd.DataFrame, out_path: Path) -> None:
    with_values = successful_series(df, "with", "rank").astype(int)
    without_values = successful_series(df, "without", "rank").astype(int)
    all_values = np.concatenate([with_values, without_values])
    bins = np.arange(all_values.min() - 0.5, all_values.max() + 1.5, 1.0)
    ticks = rank_ticks(all_values)

    fig, ax = plt.subplots()
    ax.hist(with_values, bins=bins, color=COLORS["with"], alpha=0.72, label=LABELS["with"])
    ax.hist(
        without_values,
        bins=bins,
        color=COLORS["without"],
        alpha=0.55,
        label=LABELS["without"],
    )
    ax.set_title("Bunny SDP Rank Histogram")
    ax.set_xlabel("Numerical rank of SDP solution")
    ax.set_ylabel("Trials")
    ax.set_xticks(ticks)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    results_csv = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/test/benchmark/bunny_sdp_primal_results.csv")
    )
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_csv.parent

    configure_style()
    df = pd.read_csv(results_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix in ("with", "without"):
        save_gap_hist(df, prefix, output_dir / f"bunny_relative_gap_{prefix}_redundant_hist")
        save_rank_hist(df, prefix, output_dir / f"bunny_rank_{prefix}_redundant_hist")
        save_runtime_hist(df, prefix, output_dir / f"bunny_runtime_{prefix}_redundant_hist")

    save_gap_overlay_hist(df, output_dir / "bunny_relative_gap_overlay_hist")
    save_rank_overlay_hist(df, output_dir / "bunny_rank_overlay_hist")
    save_runtime_overlay_hist(df, output_dir / "bunny_runtime_overlay_hist")

    for prefix in ("with", "without"):
        print(f"Wrote {(output_dir / f'bunny_relative_gap_{prefix}_redundant_hist').with_suffix('.png')}")
        print(f"Wrote {(output_dir / f'bunny_relative_gap_{prefix}_redundant_hist').with_suffix('.pdf')}")
        print(f"Wrote {(output_dir / f'bunny_rank_{prefix}_redundant_hist').with_suffix('.png')}")
        print(f"Wrote {(output_dir / f'bunny_rank_{prefix}_redundant_hist').with_suffix('.pdf')}")
        print(f"Wrote {(output_dir / f'bunny_runtime_{prefix}_redundant_hist').with_suffix('.png')}")
        print(f"Wrote {(output_dir / f'bunny_runtime_{prefix}_redundant_hist').with_suffix('.pdf')}")
    print(f"Wrote {(output_dir / 'bunny_relative_gap_overlay_hist').with_suffix('.png')}")
    print(f"Wrote {(output_dir / 'bunny_relative_gap_overlay_hist').with_suffix('.pdf')}")
    print(f"Wrote {(output_dir / 'bunny_rank_overlay_hist').with_suffix('.png')}")
    print(f"Wrote {(output_dir / 'bunny_rank_overlay_hist').with_suffix('.pdf')}")
    print(f"Wrote {(output_dir / 'bunny_runtime_overlay_hist').with_suffix('.png')}")
    print(f"Wrote {(output_dir / 'bunny_runtime_overlay_hist').with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
