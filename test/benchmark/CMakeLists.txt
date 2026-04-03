project(teaser_benchmark)
include(GoogleTest)

# Executable for running benchmarks
add_executable(all_benchmarks
        main.cc
        registration-benchmark.cc)
target_link_libraries(all_benchmarks
        Eigen3::Eigen
        ${TEASERPP_BLAS_LAPACK_LIBS}
        gtest
        gmock
        teaser_io
        teaser_registration
        test_tools)
gtest_add_tests(TARGET      all_benchmarks
                TEST_LIST   allBenchmarks)
set_tests_properties(${allBenchmarks}   PROPERTIES TIMEOUT 600)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(all_benchmarks OpenMP::OpenMP_CXX)
endif()

add_executable(certification_redundant_constraints_benchmark
        certification-redundant-constraints-benchmark.cc)
target_link_libraries(certification_redundant_constraints_benchmark
        Eigen3::Eigen
        ${TEASERPP_BLAS_LAPACK_LIBS}
        teaser_registration
        test_tools)
if(OpenMP_CXX_FOUND)
    target_link_libraries(certification_redundant_constraints_benchmark OpenMP::OpenMP_CXX)
endif()

add_executable(certification_redundant_constraints_montecarlo
        certification-redundant-constraints-montecarlo.cc)
target_link_libraries(certification_redundant_constraints_montecarlo
        Eigen3::Eigen
        ${TEASERPP_BLAS_LAPACK_LIBS}
        teaser_registration)
if(OpenMP_CXX_FOUND)
    target_link_libraries(certification_redundant_constraints_montecarlo OpenMP::OpenMP_CXX)
endif()

# Copy test data files to binary directory
file(COPY .
    DESTINATION .
    FILES_MATCHING
        PATTERN *.ply
        PATTERN *.label
        PATTERN *.pcd
        PATTERN *.csv
        PATTERN *.txt)
