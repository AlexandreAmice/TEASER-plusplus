#!/usr/bin/env python3

from __future__ import annotations

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


def load_bunny_points(path: Path) -> np.ndarray:
    points = []
    in_data = False
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not in_data:
                if line == "DATA ascii":
                    in_data = True
                continue
            if not line:
                continue
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.asarray(points, dtype=float).T


def sample_columns(points: np.ndarray, num_vectors: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(points.shape[1])[:num_vectors]
    return points[:, idx]


def solve_sdp(q_cost: np.ndarray, num_vectors: int, use_redundant: bool, solver: str) -> tuple[np.ndarray, float, str, float]:
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
    return z.value, float(problem.value), problem.status, runtime_ms


def numerical_rank(z: np.ndarray, tol: float = 1e-6) -> int:
    eigvals = np.linalg.eigvalsh(0.5 * (z + z.T))
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals[-1] <= 0:
      return 0
    return int(np.sum(eigvals > tol * eigvals[-1]))


def run_trial(points: np.ndarray, rng: np.random.Generator, num_vectors: int, outlier_ratio: float, noise_bound: float, solver: str) -> dict:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    v1 = sample_columns(points, num_vectors, rng)
    v2 = r @ v1 + rng.uniform(-noise_bound, noise_bound, size=(3, num_vectors))

    num_outliers = int(round(num_vectors * outlier_ratio))
    if num_outliers > 0:
        idxs = np.arange(num_vectors - num_outliers, num_vectors)
        v2[:, idxs] = rng.uniform(3.0, 8.0, size=(3, num_outliers))

    q_cost = get_q_cost(v1, v2, noise_bound)
    z_with, obj_with, status_with, runtime_with = solve_sdp(q_cost, num_vectors, True, solver)
    z_without, obj_without, status_without, runtime_without = solve_sdp(q_cost, num_vectors, False, solver)

    return {
        "with_rank": numerical_rank(z_with),
        "without_rank": numerical_rank(z_without),
        "with_objective": obj_with,
        "without_objective": obj_without,
        "with_status": status_with,
        "without_status": status_without,
        "with_runtime_ms": runtime_with,
        "without_runtime_ms": runtime_without,
    }


def main() -> int:
    out_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/test/benchmark/sdp_rank_distribution.csv")
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    num_vectors = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    outlier_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    noise_bound = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
    solver = sys.argv[6] if len(sys.argv) > 6 else "SCS"
    bunny_path = Path(sys.argv[7]) if len(sys.argv) > 7 else Path("test/teaser/data/bunny.pcd")

    rng = np.random.default_rng(42)
    points = load_bunny_points(bunny_path)
    rows = []
    for trial in range(num_trials):
        row = {
            "trial": trial,
            "num_vectors": num_vectors,
            "outlier_ratio": outlier_ratio,
            "noise_bound": noise_bound,
        }
        row.update(run_trial(points, rng, num_vectors, outlier_ratio, noise_bound, solver))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
