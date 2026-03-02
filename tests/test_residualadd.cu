#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/residual_add.cuh"

TEST(ResidualAdd, MatchesPyTorch) {
    std::vector<half> input = load_tensor("test_data/residual_add_input.bin");
    std::vector<half> delta = load_tensor("test_data/residual_add_delta.bin");
    std::vector<half> expected = load_tensor("test_data/residual_add_output.bin");
    std::vector<half> actual(expected.size());

    const int rows = 128;
    const int cols = InfEngineConfig::HIDDEN_SIZE;
    EXPECT_EQ(rows * cols, input.size());
    EXPECT_EQ(rows * cols, delta.size());
    EXPECT_EQ(rows * cols, expected.size());

    half* d_input;
    half* d_delta;
    cudaMalloc(&d_input, sizeof(half) * input.size());
    cudaMalloc(&d_delta, sizeof(half) * delta.size());
    cudaMemcpy(d_input, input.data(), sizeof(half) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, delta.data(), sizeof(half) * delta.size(), cudaMemcpyHostToDevice);

    residual_add(rows, cols, d_input, d_delta);

    // Copy d_input back since residual_add is in-place (writes result into d_input)
    cudaMemcpy(actual.data(), d_input, sizeof(half) * input.size(), cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 1e-5);

    EXPECT_LT(result.max_abs_err, 1e-3);
    EXPECT_GT(result.pct_within_tol, 99.0);

    cudaFree(d_input);
    cudaFree(d_delta);
}