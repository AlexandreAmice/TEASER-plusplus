/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "teaser/certification.h"
#include "test_utils.h"

namespace {

struct CertificationCase {
  std::string name;
  teaser::DRSCertifier::Params params;
  Eigen::Matrix<double, 3, Eigen::Dynamic> v1;
  Eigen::Matrix<double, 3, Eigen::Dynamic> v2;
  Eigen::Matrix3d R_est;
  Eigen::Matrix<double, 1, Eigen::Dynamic> theta_est;
};

struct RunStats {
  bool is_optimal = false;
  double best_suboptimality = std::numeric_limits<double>::infinity();
  double runtime_ms = 0;
  size_t iterations = 0;
};

struct AggregateStats {
  size_t num_cases = 0;
  size_t num_certified = 0;
  double best_suboptimality_sum = 0;
  double runtime_sum_ms = 0;
};

bool fileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

std::string findCertificationDataRoot(int argc, char** argv) {
  if (argc > 1 && fileExists(std::string(argv[1]) + "/certification_small_instances/case_1/parameters.txt")) {
    return argv[1];
  }

  const std::vector<std::string> candidates = {
      "./teaser/data",
      "../teaser/data",
      "../../test/teaser/data",
      "./test/teaser/data",
      "test/teaser/data",
  };

  for (const auto& candidate : candidates) {
    if (fileExists(candidate + "/certification_small_instances/case_1/parameters.txt")) {
      return candidate;
    }
  }

  return "";
}

void loadScalarParameters(const std::string& file_path, teaser::DRSCertifier::Params* params) {
  std::ifstream file(file_path);
  if (!file) {
    throw std::runtime_error("Unable to open file: " + file_path);
  }

  std::string line;
  const std::string delimiter = ":";
  while (std::getline(file, line)) {
    const size_t delim_idx = line.find(delimiter, 0);
    const std::string param = line.substr(0, delim_idx);
    const std::string value = line.substr(delim_idx + 2, line.length());

    if (param == "cbar2") {
      params->cbar2 = std::stod(value);
    } else if (param == "noise_bound") {
      params->noise_bound = std::stod(value);
    } else if (param == "max_iterations") {
      params->max_iterations = std::stod(value);
    }
  }
}

CertificationCase loadCase(const std::string& root, const std::string& name) {
  CertificationCase data;
  data.name = name;

  const std::string case_dir = root + "/" + name;
  loadScalarParameters(case_dir + "/parameters.txt", &(data.params));

  std::ifstream v1_source_file(case_dir + "/v1.csv");
  data.v1 = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v1_source_file);

  std::ifstream v2_source_file(case_dir + "/v2.csv");
  data.v2 = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v2_source_file);

  std::ifstream r_est_source_file(case_dir + "/R_est.csv");
  data.R_est = teaser::test::readFileToEigenFixedMatrix<double, 3, 3>(r_est_source_file);

  std::ifstream theta_source_file(case_dir + "/theta_est.csv");
  data.theta_est = teaser::test::readFileToEigenMatrix<double, 1, Eigen::Dynamic>(theta_source_file);

  return data;
}

std::vector<CertificationCase> loadCases(const std::string& root) {
  std::vector<CertificationCase> cases;
  const auto case_names = teaser::test::readSubdirs(root);
  cases.reserve(case_names.size());
  for (const auto& name : case_names) {
    cases.emplace_back(loadCase(root, name));
  }
  return cases;
}

RunStats runCase(const CertificationCase& test_case, bool use_redundant_constraints) {
  auto params = test_case.params;
  params.use_redundant_constraints = use_redundant_constraints;

  teaser::DRSCertifier certifier(params);
  std::ostringstream sink;
  auto* old_buf = std::cout.rdbuf(sink.rdbuf());

  auto start = std::chrono::steady_clock::now();
  auto result = certifier.certify(test_case.R_est, test_case.v1, test_case.v2, test_case.theta_est);
  auto stop = std::chrono::steady_clock::now();
  std::cout.rdbuf(old_buf);

  RunStats stats;
  stats.is_optimal = result.is_optimal;
  stats.best_suboptimality = result.best_suboptimality;
  stats.runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  stats.iterations = result.suboptimality_traj.size();
  return stats;
}

void updateAggregate(const RunStats& stats, AggregateStats* aggregate) {
  aggregate->num_cases += 1;
  aggregate->num_certified += static_cast<size_t>(stats.is_optimal);
  aggregate->best_suboptimality_sum += stats.best_suboptimality;
  aggregate->runtime_sum_ms += stats.runtime_ms;
}

void runDataset(const std::string& label, const std::vector<CertificationCase>& cases) {
  AggregateStats with_redundant;
  AggregateStats without_redundant;

  std::cout << "\nDataset: " << label << "\n";
  std::cout << "Case                         N   with_gap      no_gap        gap_ratio     with_ms   no_ms     with_ok no_ok\n";
  std::cout << "---------------------------------------------------------------------------------------------------------------\n";

  for (const auto& test_case : cases) {
    const auto with_stats = runCase(test_case, true);
    const auto without_stats = runCase(test_case, false);

    updateAggregate(with_stats, &with_redundant);
    updateAggregate(without_stats, &without_redundant);

    double ratio = std::numeric_limits<double>::infinity();
    if (std::isfinite(with_stats.best_suboptimality) && with_stats.best_suboptimality > 0) {
      ratio = without_stats.best_suboptimality / with_stats.best_suboptimality;
    }

    std::cout << std::left << std::setw(28) << test_case.name << std::right << std::setw(4)
              << test_case.v1.cols() << " " << std::setw(13) << std::setprecision(6)
              << std::scientific << with_stats.best_suboptimality << " " << std::setw(13)
              << without_stats.best_suboptimality << " ";
    if (std::isfinite(ratio)) {
      std::cout << std::setw(13) << ratio;
    } else {
      std::cout << std::setw(13) << "inf";
    }
    std::cout << " " << std::setw(9) << std::fixed << std::setprecision(3) << with_stats.runtime_ms
              << " " << std::setw(9) << without_stats.runtime_ms << " " << std::setw(7)
              << (with_stats.is_optimal ? "yes" : "no") << " " << std::setw(5)
              << (without_stats.is_optimal ? "yes" : "no") << "\n";
  }

  const auto mean_gap_with = with_redundant.best_suboptimality_sum / with_redundant.num_cases;
  const auto mean_gap_without =
      without_redundant.best_suboptimality_sum / without_redundant.num_cases;
  const auto mean_ms_with = with_redundant.runtime_sum_ms / with_redundant.num_cases;
  const auto mean_ms_without = without_redundant.runtime_sum_ms / without_redundant.num_cases;

  std::cout << "---------------------------------------------------------------------------------------------------------------\n";
  std::cout << "Summary\n";
  std::cout << "  certified with redundant constraints   : " << with_redundant.num_certified << "/"
            << with_redundant.num_cases << "\n";
  std::cout << "  certified without redundant constraints: " << without_redundant.num_certified << "/"
            << without_redundant.num_cases << "\n";
  std::cout << "  mean best suboptimality                : " << std::scientific << mean_gap_with
            << " vs " << mean_gap_without << " (with vs without)\n";
  std::cout << "  mean runtime (ms)                      : " << std::fixed << std::setprecision(3)
            << mean_ms_with << " vs " << mean_ms_without << " (with vs without)\n";
}

} // namespace

int main(int argc, char** argv) {
  const std::string data_root = findCertificationDataRoot(argc, argv);
  if (data_root.empty()) {
    std::cerr << "Could not locate certification benchmark data.\n";
    std::cerr << "Pass the data root explicitly, e.g. ./certification_redundant_constraints_benchmark "
                 "test/teaser/data\n";
    return 1;
  }

  std::cout << "Certification ablation for redundant/implied SDP constraints\n";
  std::cout << "Note: this repo does not ship the original CVX-based SDP solver from the paper.\n";
  std::cout << "This benchmark removes the certifier-side dual contribution induced by the redundant\n";
  std::cout << "off-diagonal symmetry constraints and compares the resulting certification quality.\n";
  std::cout << "Using data root: " << data_root << "\n";

  runDataset("small", loadCases(data_root + "/certification_small_instances"));
  runDataset("large", loadCases(data_root + "/certification_large_instances"));

  return 0;
}
