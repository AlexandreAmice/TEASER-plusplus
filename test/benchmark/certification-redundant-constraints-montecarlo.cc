/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "teaser/certification.h"

namespace {

struct RunStats {
  bool is_optimal = false;
  double best_suboptimality = std::numeric_limits<double>::infinity();
  double runtime_ms = 0;
  size_t iterations = 0;
};

struct TrialConfig {
  int num_trials = 40;
  int num_vectors = 20;
  double outlier_ratio = 0.2;
  double noise_bound = 0.01;
  int max_iterations = 100;
};

RunStats runCertifier(const teaser::DRSCertifier::Params& params, const Eigen::Matrix3d& R_est,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2,
                      const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_est) {
  teaser::DRSCertifier certifier(params);
  std::ostringstream sink;
  auto* old_buf = std::cout.rdbuf(sink.rdbuf());

  auto start = std::chrono::steady_clock::now();
  const auto result = certifier.certify(R_est, v1, v2, theta_est);
  auto stop = std::chrono::steady_clock::now();
  std::cout.rdbuf(old_buf);

  RunStats stats;
  stats.is_optimal = result.is_optimal;
  stats.best_suboptimality = result.best_suboptimality;
  stats.runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  stats.iterations = result.suboptimality_traj.size();
  return stats;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> uniformRandomMatrix3xN(int cols, std::mt19937* rng,
                                                                double low, double high) {
  std::uniform_real_distribution<double> dist(low, high);
  Eigen::Matrix<double, 3, Eigen::Dynamic> mat(3, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < 3; ++r) {
      mat(r, c) = dist(*rng);
    }
  }
  return mat;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> boundedNoise3xN(int cols, std::mt19937* rng,
                                                         double bound) {
  return uniformRandomMatrix3xN(cols, rng, -bound, bound);
}

} // namespace

int main(int argc, char** argv) {
  TrialConfig config;
  const std::string output_csv =
      argc > 1 ? argv[1] : "build/test/benchmark/certification_redundant_constraints_montecarlo.csv";
  if (argc > 2) {
    config.num_trials = std::stoi(argv[2]);
  }
  if (argc > 3) {
    config.num_vectors = std::stoi(argv[3]);
  }
  if (argc > 4) {
    config.outlier_ratio = std::stod(argv[4]);
  }
  if (argc > 5) {
    config.max_iterations = std::stoi(argv[5]);
  }

  std::ofstream csv(output_csv);
  if (!csv) {
    std::cerr << "Unable to open output CSV: " << output_csv << "\n";
    return 1;
  }

  csv << "trial,num_vectors,outlier_ratio,with_gap,without_gap,with_runtime_ms,without_runtime_ms,"
         "with_iterations,without_iterations,with_certified,without_certified\n";

  teaser::DRSCertifier::Params params;
  params.cbar2 = 1;
  params.noise_bound = config.noise_bound;
  params.max_iterations = config.max_iterations;

  std::mt19937 rng(42);
  const int num_outliers = static_cast<int>(std::round(config.num_vectors * config.outlier_ratio));

  for (int trial = 0; trial < config.num_trials; ++trial) {
    const auto q_gt = Eigen::Quaterniond::UnitRandom();
    const auto R_gt = q_gt.toRotationMatrix();

    Eigen::Matrix<double, 3, Eigen::Dynamic> v1 =
        uniformRandomMatrix3xN(config.num_vectors, &rng, -1.0, 1.0);
    Eigen::Matrix<double, 3, Eigen::Dynamic> v2 = R_gt * v1 + boundedNoise3xN(config.num_vectors, &rng, config.noise_bound);

    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_est =
        Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, config.num_vectors);

    for (int idx = config.num_vectors - num_outliers; idx < config.num_vectors; ++idx) {
      if (idx < 0) {
        continue;
      }
      v2.col(idx) = uniformRandomMatrix3xN(1, &rng, 3.0, 8.0);
      theta_est(idx) = -1.0;
    }

    auto with_params = params;
    with_params.use_redundant_constraints = true;
    const auto with_stats = runCertifier(with_params, R_gt, v1, v2, theta_est);

    auto without_params = params;
    without_params.use_redundant_constraints = false;
    const auto without_stats = runCertifier(without_params, R_gt, v1, v2, theta_est);

    csv << trial << "," << config.num_vectors << "," << config.outlier_ratio << ","
        << with_stats.best_suboptimality << "," << without_stats.best_suboptimality << ","
        << with_stats.runtime_ms << "," << without_stats.runtime_ms << ","
        << with_stats.iterations << "," << without_stats.iterations << ","
        << static_cast<int>(with_stats.is_optimal) << ","
        << static_cast<int>(without_stats.is_optimal) << "\n";
    csv.flush();
  }

  std::cout << "Wrote Monte Carlo ablation results to " << output_csv << "\n";
  return 0;
}
