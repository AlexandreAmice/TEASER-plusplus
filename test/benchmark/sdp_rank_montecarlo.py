#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


P = np.array(
    [
        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0],
        [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0],
        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)

VALID_STATUSES = {"optimal", "optimal_inaccurate"}


def get_q_cost(v1: np.ndarray, v2: np.ndarray, noise_bound: float, cbar2: float = 1.0) -> np.ndarray:
    n = v1.shape[1]
    npm = 4 + 4 * n
    noise_bound_scaled = cbar2 * (noise_bound ** 2)

    q1 = np.zeros((npm, npm))
    q2 = np.zeros((npm, npm))

    for k in range(n):
        start = 4 + 4 * k
        temp_a = np.outer(v2[:, k], v1[:, k])
        temp_map = temp_a.reshape(9, order="F")
        temp_b = P.T @ temp_map
        p_k = temp_b.reshape((4, 4), order="F")

        ck1 = 0.5 * (np.dot(v1[:, k], v1[:, k]) + np.dot(v2[:, k], v2[:, k]) - noise_bound_scaled)
        q1[0:4, start : start + 4] += -0.5 * p_k + 0.5 * ck1 * np.eye(4)
        q1[start : start + 4, 0:4] += -0.5 * p_k + 0.5 * ck1 * np.eye(4)

        ck2 = 0.5 * (np.dot(v1[:, k], v1[:, k]) + np.dot(v2[:, k], v2[:, k]) + noise_bound_scaled)
        q2[start : start + 4, start : start + 4] += -p_k + ck2 * np.eye(4)

    return q1 + q2


def solve_sdp(
    q_cost: np.ndarray,
    num_vectors: int,
    use_redundant: bool,
    solver: str,
) -> tuple[np.ndarray | None, float, str, float]:
    npm = 4 + 4 * num_vectors
    z = cp.Variable((npm, npm), symmetric=True)
    constraints = [z >> 0, cp.trace(z[0:4, 0:4]) == 1]

    z00 = z[0:4, 0:4]
    for k in range(num_vectors):
        start = 4 + 4 * k
        constraints.append(z[start : start + 4, start : start + 4] == z00)

    if use_redundant:
        for i in range(num_vectors + 1):
            for j in range(i + 1, num_vectors + 1):
                bi = slice(4 * i, 4 * i + 4)
                bj = slice(4 * j, 4 * j + 4)
                constraints.append(z[bi, bj] == cp.transpose(z[bi, bj]))

    problem = cp.Problem(cp.Minimize(cp.trace(q_cost @ z)), constraints)
    start = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    value = float(problem.value) if problem.value is not None else float("nan")
    return z.value, value, str(problem.status), runtime_ms


def numerical_rank(z: np.ndarray | None, tol: float = 1e-6) -> float:
    if z is None or not np.all(np.isfinite(z)):
        return float("nan")
    eigvals = np.linalg.eigvalsh(0.5 * (z + z.T))
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals[-1] <= 0:
        return 0.0
    return float(np.sum(eigvals > tol * eigvals[-1]))


def build_candidate_cost(q_cost: np.ndarray, q: np.ndarray, theta: np.ndarray) -> float:
    theta_prepended = np.concatenate(([1.0], theta))
    x_hat = np.kron(theta_prepended, q)
    return float(x_hat @ q_cost @ x_hat)


def relative_gap(candidate_cost: float, optimum_cost: float) -> float:
    eps = 1e-12
    denom = max(abs(optimum_cost), eps)
    return float((candidate_cost - optimum_cost) / denom)


def summarize_condition(df: pd.DataFrame, prefix: str) -> dict[str, object]:
    valid = df[df[f"{prefix}_status"].isin(VALID_STATUSES)].copy()
    summary: dict[str, object] = {
        "condition": prefix,
        "num_total_trials": int(len(df)),
        "num_successful_solves": int(len(valid)),
    }
    if valid.empty:
        summary.update(
            {
                "median_relative_gap": float("nan"),
                "q25_relative_gap": float("nan"),
                "q75_relative_gap": float("nan"),
                "median_rank": float("nan"),
                "q25_rank": float("nan"),
                "q75_rank": float("nan"),
                "median_runtime_ms": float("nan"),
                "q25_runtime_ms": float("nan"),
                "q75_runtime_ms": float("nan"),
                "rank_counts": json.dumps({}),
            }
        )
        return summary

    gap = valid[f"{prefix}_relative_gap"].to_numpy()
    rank = valid[f"{prefix}_rank"].to_numpy()
    runtime = valid[f"{prefix}_runtime_ms"].to_numpy()
    unique_rank, counts = np.unique(rank.astype(int), return_counts=True)
    rank_counts = {int(r): int(c) for r, c in zip(unique_rank, counts)}

    summary.update(
        {
            "median_relative_gap": float(np.median(gap)),
            "q25_relative_gap": float(np.quantile(gap, 0.25)),
            "q75_relative_gap": float(np.quantile(gap, 0.75)),
            "median_rank": float(np.median(rank)),
            "q25_rank": float(np.quantile(rank, 0.25)),
            "q75_rank": float(np.quantile(rank, 0.75)),
            "median_runtime_ms": float(np.median(runtime)),
            "q25_runtime_ms": float(np.quantile(runtime, 0.25)),
            "q75_runtime_ms": float(np.quantile(runtime, 0.75)),
            "rank_counts": json.dumps(rank_counts, sort_keys=True),
        }
    )
    return summary


def main() -> int:
    out_results_csv = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/test/benchmark/bunny_sdp_primal_results.csv")
    )
    out_summary_csv = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("build/test/benchmark/bunny_sdp_summary.csv")
    )
    trial_summary_csv = (
        Path(sys.argv[3]) if len(sys.argv) > 3 else Path("build/test/benchmark/bunny_teaser_candidate_trials.csv")
    )
    tims_csv = (
        Path(sys.argv[4]) if len(sys.argv) > 4 else Path("build/test/benchmark/bunny_teaser_candidate_tims.csv")
    )
    solver = sys.argv[5] if len(sys.argv) > 5 else "CLARABEL"

    trials_df = pd.read_csv(trial_summary_csv)
    tims_df = pd.read_csv(tims_csv)

    rows: list[dict[str, object]] = []
    for trial in trials_df.itertuples(index=False):
        row: dict[str, object] = {
            "trial": int(trial.trial),
            "num_points": int(trial.num_points),
            "outlier_ratio": float(trial.outlier_ratio),
            "noise_bound": float(trial.noise_bound),
            "valid": int(trial.valid),
            "teaser_runtime_ms": float(trial.teaser_runtime_ms),
            "num_rotation_tims": int(trial.num_rotation_tims),
            "rotation_inlier_count": int(trial.rotation_inlier_count),
        }

        if int(trial.valid) != 1:
            row.update(
                {
                    "candidate_cost": float("nan"),
                    "with_objective": float("nan"),
                    "without_objective": float("nan"),
                    "with_relative_gap": float("nan"),
                    "without_relative_gap": float("nan"),
                    "with_rank": float("nan"),
                    "without_rank": float("nan"),
                    "with_status": "invalid_teaser_solution",
                    "without_status": "invalid_teaser_solution",
                    "with_runtime_ms": float("nan"),
                    "without_runtime_ms": float("nan"),
                }
            )
            rows.append(row)
            continue

        trial_tims = tims_df[tims_df["trial"] == int(trial.trial)].sort_values("tim_index")
        v1 = trial_tims[["v1x", "v1y", "v1z"]].to_numpy(dtype=float).T
        v2 = trial_tims[["v2x", "v2y", "v2z"]].to_numpy(dtype=float).T
        theta = trial_tims["theta"].to_numpy(dtype=float)
        q = np.array([trial.qx, trial.qy, trial.qz, trial.qw], dtype=float)
        q /= np.linalg.norm(q)

        q_cost = get_q_cost(v1, v2, float(trial.noise_bound))
        candidate_cost = build_candidate_cost(q_cost, q, theta)

        z_with, obj_with, status_with, runtime_with = solve_sdp(
            q_cost, int(trial.num_rotation_tims), True, solver
        )
        z_without, obj_without, status_without, runtime_without = solve_sdp(
            q_cost, int(trial.num_rotation_tims), False, solver
        )

        row.update(
            {
                "candidate_cost": candidate_cost,
                "with_objective": obj_with,
                "without_objective": obj_without,
                "with_relative_gap": relative_gap(candidate_cost, obj_with)
                if status_with in VALID_STATUSES
                else float("nan"),
                "without_relative_gap": relative_gap(candidate_cost, obj_without)
                if status_without in VALID_STATUSES
                else float("nan"),
                "with_rank": numerical_rank(z_with) if status_with in VALID_STATUSES else float("nan"),
                "without_rank": numerical_rank(z_without)
                if status_without in VALID_STATUSES
                else float("nan"),
                "with_status": status_with,
                "without_status": status_without,
                "with_runtime_ms": runtime_with,
                "without_runtime_ms": runtime_without,
            }
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [summarize_condition(results_df, "with"), summarize_condition(results_df, "without")]
    )

    out_results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_results_csv, index=False)
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Wrote {out_results_csv}")
    print(f"Wrote {out_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
