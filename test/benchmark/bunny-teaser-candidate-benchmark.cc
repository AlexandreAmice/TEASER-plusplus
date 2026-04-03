/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "teaser/registration.h"

namespace {

using Matrix3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;

struct TrialInputs {
  Matrix3X src_points;
  Matrix3X dst_points;
};

Eigen::Quaterniond sampleQuaternion(std::mt19937* rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::Vector4d q;
  q << normal(*rng), normal(*rng), normal(*rng), normal(*rng);
  q.normalize();
  return Eigen::Quaterniond(q(3), q(0), q(1), q(2));
}

Matrix3X loadBunnyPoints(const std::string& path) {
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

  Matrix3X mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    mat.col(i) = points[i];
  }
  return mat;
}

Matrix3X sampleColumns(const Matrix3X& points, int sample_size, std::mt19937* rng) {
  std::vector<int> indices(points.cols());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), *rng);

  Matrix3X sampled(3, sample_size);
  for (int i = 0; i < sample_size; ++i) {
    sampled.col(i) = points.col(indices[i]);
  }
  return sampled;
}

TrialInputs generateTrial(const Matrix3X& bunny, int num_points, double outlier_ratio,
                          double noise_bound, std::mt19937* rng) {
  TrialInputs trial;
  trial.src_points = sampleColumns(bunny, num_points, rng);

  const auto q_gt = sampleQuaternion(rng);
  const auto rotation_gt = q_gt.toRotationMatrix();
  trial.dst_points = rotation_gt * trial.src_points;

  std::uniform_real_distribution<double> noise_dist(-noise_bound, noise_bound);
  for (int c = 0; c < trial.dst_points.cols(); ++c) {
    for (int r = 0; r < 3; ++r) {
      trial.dst_points(r, c) += noise_dist(*rng);
    }
  }

  const int num_outliers = static_cast<int>(std::round(num_points * outlier_ratio));
  for (int i = num_points - num_outliers; i < num_points; ++i) {
    if (i < 0) {
      continue;
    }
    trial.dst_points.col(i).setRandom();
    trial.dst_points.col(i) = 5.0 * trial.dst_points.col(i).array() + 5.0;
  }

  return trial;
}

teaser::RobustRegistrationSolver::Params makeSolverParams(double noise_bound) {
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = noise_bound;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 1e-12;
  params.rotation_tim_graph =
      teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
  params.inlier_selection_mode =
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
  return params;
}

void writeInvalidTrial(std::ofstream* summary_csv, int trial, int num_points, double outlier_ratio,
                       double noise_bound, double runtime_ms) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  *summary_csv << trial << "," << num_points << "," << outlier_ratio << "," << noise_bound << ",0,"
               << runtime_ms << ",0,0," << nan << "," << nan << "," << nan << "," << nan << "\n";
}

} // namespace

int main(int argc, char** argv) {
  const std::string summary_csv_path =
      argc > 1 ? argv[1] : "build/test/benchmark/bunny_teaser_candidate_trials.csv";
  const std::string tims_csv_path =
      argc > 2 ? argv[2] : "build/test/benchmark/bunny_teaser_candidate_tims.csv";
  const int trials = argc > 3 ? std::stoi(argv[3]) : 100;
  const int num_points = argc > 4 ? std::stoi(argv[4]) : 20;
  const double outlier_ratio = argc > 5 ? std::stod(argv[5]) : 0.2;
  const double noise_bound = argc > 6 ? std::stod(argv[6]) : 0.01;
  const std::string bunny_path = argc > 7 ? argv[7] : "test/teaser/data/bunny.pcd";

  std::ofstream summary_csv(summary_csv_path);
  if (!summary_csv) {
    std::cerr << "Unable to open output CSV: " << summary_csv_path << "\n";
    return 1;
  }

  std::ofstream tims_csv(tims_csv_path);
  if (!tims_csv) {
    std::cerr << "Unable to open TIM CSV: " << tims_csv_path << "\n";
    return 1;
  }

  summary_csv << "trial,num_points,outlier_ratio,noise_bound,valid,teaser_runtime_ms,num_rotation_tims,"
                 "rotation_inlier_count,qx,qy,qz,qw\n";
  tims_csv << "trial,tim_index,theta,v1x,v1y,v1z,v2x,v2y,v2z\n";

  const auto bunny = loadBunnyPoints(bunny_path);
  std::mt19937 rng(42);
  const auto params = makeSolverParams(noise_bound);

  for (int trial_idx = 0; trial_idx < trials; ++trial_idx) {
    auto trial = generateTrial(bunny, num_points, outlier_ratio, noise_bound, &rng);

    teaser::RobustRegistrationSolver solver(params);
    std::ostringstream sink;
    auto* old_buf = std::cout.rdbuf(sink.rdbuf());
    auto start = std::chrono::steady_clock::now();
    const auto solution = solver.solve(trial.src_points, trial.dst_points);
    auto stop = std::chrono::steady_clock::now();
    std::cout.rdbuf(old_buf);
    const double runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    if (!solution.valid) {
      writeInvalidTrial(&summary_csv, trial_idx, num_points, outlier_ratio, noise_bound,
                        runtime_ms);
      continue;
    }

    const auto src_tims = solver.getMaxCliqueSrcTIMs();
    const auto dst_tims = solver.getMaxCliqueDstTIMs();
    const auto theta_mask = solver.getRotationInliersMask();

    if (src_tims.cols() == 0 || dst_tims.cols() == 0 || theta_mask.cols() != src_tims.cols()) {
      writeInvalidTrial(&summary_csv, trial_idx, num_points, outlier_ratio, noise_bound,
                        runtime_ms);
      continue;
    }

    const auto rotation_inlier_count =
        static_cast<int>(theta_mask.cast<int>().row(0).sum());
    Eigen::Quaterniond q(solution.rotation);
    q.normalize();

    summary_csv << trial_idx << "," << num_points << "," << outlier_ratio << "," << noise_bound
                << ",1," << runtime_ms << "," << src_tims.cols() << "," << rotation_inlier_count
                << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";

    for (int tim_idx = 0; tim_idx < src_tims.cols(); ++tim_idx) {
      const int theta = theta_mask(0, tim_idx) ? 1 : -1;
      tims_csv << trial_idx << "," << tim_idx << "," << theta << "," << src_tims(0, tim_idx)
               << "," << src_tims(1, tim_idx) << "," << src_tims(2, tim_idx) << ","
               << dst_tims(0, tim_idx) << "," << dst_tims(1, tim_idx) << ","
               << dst_tims(2, tim_idx) << "\n";
    }

    summary_csv.flush();
    tims_csv.flush();
  }

  std::cout << "Wrote " << summary_csv_path << "\n";
  std::cout << "Wrote " << tims_csv_path << "\n";
  return 0;
}
