/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

Eigen::Matrix<double, 3, Eigen::Dynamic> loadBunnyPoints(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Unable to open bunny PCD file: " + path);
  }

  std::string line;
  bool found_data = false;
  std::vector<Eigen::Vector3d> points;
  while (std::getline(file, line)) {
    if (!found_data) {
      if (line == "DATA ascii") {
        found_data = true;
      }
      continue;
    }
    if (line.empty()) {
      continue;
    }
    std::istringstream ss(line);
    Eigen::Vector3d p;
    ss >> p.x() >> p.y() >> p.z();
    points.push_back(p);
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    mat.col(i) = points[i];
  }
  return mat;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> sampleColumns(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& points, int sample_size, std::mt19937* rng) {
  std::vector<int> indices(points.cols());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), *rng);

  Eigen::Matrix<double, 3, Eigen::Dynamic> sampled(3, sample_size);
  for (int i = 0; i < sample_size; ++i) {
    sampled.col(i) = points.col(indices[i]);
  }
  return sampled;
}

RunStats runCertifier(const teaser::DRSCertifier::Params& params, const Eigen::Matrix3d& rotation,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                      const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta) {
  teaser::DRSCertifier certifier(params);
  std::ostringstream sink;
  auto* old_buf = std::cout.rdbuf(sink.rdbuf());

  auto start = std::chrono::steady_clock::now();
  const auto result = certifier.certify(rotation, src, dst, theta);
  auto stop = std::chrono::steady_clock::now();

  std::cout.rdbuf(old_buf);

  RunStats stats;
  stats.is_optimal = result.is_optimal;
  stats.best_suboptimality = result.best_suboptimality;
  stats.runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  stats.iterations = result.suboptimality_traj.size();
  return stats;
}

} // namespace

int main(int argc, char** argv) {
  const std::string output_csv =
      argc > 1 ? argv[1] : "build/test/benchmark/bunny_certifier_suboptimality.csv";
  const int trials = argc > 2 ? std::stoi(argv[2]) : 20;
  const int num_points = argc > 3 ? std::stoi(argv[3]) : 12;
  const double outlier_ratio = argc > 4 ? std::stod(argv[4]) : 0.2;
  const double noise_bound = argc > 5 ? std::stod(argv[5]) : 0.01;
  const int max_iters = argc > 6 ? std::stoi(argv[6]) : 50;
  const std::string bunny_path =
      argc > 7 ? argv[7] : "test/teaser/data/bunny.pcd";

  std::ofstream csv(output_csv);
  if (!csv) {
    std::cerr << "Unable to open output CSV: " << output_csv << "\n";
    return 1;
  }

  csv << "trial,num_points,outlier_ratio,noise_bound,with_gap,without_gap,with_runtime_ms,"
         "without_runtime_ms,with_iterations,without_iterations,with_certified,without_certified\n";

  const auto bunny = loadBunnyPoints(bunny_path);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> noise_dist(-noise_bound, noise_bound);
  const int num_outliers = static_cast<int>(std::round(num_points * outlier_ratio));

  teaser::DRSCertifier::Params with_params;
  with_params.cbar2 = 1;
  with_params.noise_bound = noise_bound;
  with_params.max_iterations = max_iters;
  with_params.use_redundant_constraints = true;

  auto without_params = with_params;
  without_params.use_redundant_constraints = false;

  for (int trial = 0; trial < trials; ++trial) {
    const auto src = sampleColumns(bunny, num_points, &rng);
    const auto q = Eigen::Quaterniond::UnitRandom();
    const auto rotation = q.toRotationMatrix();

    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = rotation * src;
    for (int c = 0; c < dst.cols(); ++c) {
      for (int r = 0; r < 3; ++r) {
        dst(r, c) += noise_dist(rng);
      }
    }

    Eigen::Matrix<double, 1, Eigen::Dynamic> theta =
        Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, num_points);
    for (int i = num_points - num_outliers; i < num_points; ++i) {
      if (i < 0) {
        continue;
      }
      dst.col(i).setRandom();
      dst.col(i) = 5.0 * dst.col(i).array() + 5.0;
      theta(i) = -1.0;
    }

    const auto with_stats = runCertifier(with_params, rotation, src, dst, theta);
    const auto without_stats = runCertifier(without_params, rotation, src, dst, theta);

    csv << trial << "," << num_points << "," << outlier_ratio << "," << noise_bound << ","
        << with_stats.best_suboptimality << "," << without_stats.best_suboptimality << ","
        << with_stats.runtime_ms << "," << without_stats.runtime_ms << ","
        << with_stats.iterations << "," << without_stats.iterations << ","
        << static_cast<int>(with_stats.is_optimal) << ","
        << static_cast<int>(without_stats.is_optimal) << "\n";
    csv.flush();
  }

  std::cout << "Wrote " << output_csv << "\n";
  return 0;
}
