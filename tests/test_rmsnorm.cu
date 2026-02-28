#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/rmsnorm.cuh"

TEST(RMSNorm, MatchesPyTorch) {
    std::vector<__half> input = load_tensor("test_data/rmsnorm_input.bin");
    std::vector<__half> weight = load_tensor("test_data/rmsnorm_weight.bin");
    std::vector<__half> expected = load_tensor("test_data/rmsnorm_output.bin");
    std::vector<__half> actual(expected.size());

    int seq_len = input.size() / InfEngineConfig::HIDDEN_SIZE;
    const int output_size_bytes = InfEngineConfig::HALF_SIZE * expected.size();

    __half* d_input;
    __half* d_weight;
    __half* d_actual;
    cudaMalloc(&d_input, InfEngineConfig::HALF_SIZE * input.size());
    cudaMemcpy(d_input, input.data(), InfEngineConfig::HALF_SIZE * input.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weight, InfEngineConfig::HALF_SIZE * weight.size());
    cudaMemcpy(d_weight, weight.data(), InfEngineConfig::HALF_SIZE * weight.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_actual, output_size_bytes);

    // needs to be launched with 
    rmsnorm(seq_len, d_input, d_weight, d_actual);
    cudaMemcpy(actual.data(), d_actual, output_size_bytes, cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size());

    EXPECT_LT(result.max_abs_err, 1e-3);
    EXPECT_GT(result.pct_within_tol, 99.0);

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_actual);
}