#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/softmax.cuh"

TEST(ScaleSoftmax, MatchesPyTorch) {
    // scalesoftmax takes [q_heads, seq_len, seq_len] -> [q_heads, seq_len, seq_len]
    std::vector<__half> input = load_tensor("test_data/scalesoftmax_input.bin");
    std::vector<__half> expected = load_tensor("test_data/scalesoftmax_output.bin");
    std::vector<__half> actual(expected.size());

    const int seq_len = static_cast<int>(std::round(std::sqrt(input.size() / InfEngineConfig::NUM_Q_HEADS)));
    const int output_size_bytes = InfEngineConfig::HALF_SIZE * expected.size();

    __half* d_input;
    __half* d_actual;

    cudaMalloc(&d_input, InfEngineConfig::HALF_SIZE * input.size());
    cudaMemcpy(d_input, input.data(), InfEngineConfig::HALF_SIZE * input.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_actual, output_size_bytes);

    scale_causal_softmax(seq_len, d_input, d_actual);
    cudaMemcpy(actual.data(), d_actual, output_size_bytes, cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 2e-5);

    EXPECT_LT(result.max_abs_err, 1e-3);
    EXPECT_GT(result.pct_within_tol, 99.0);

    cudaFree(d_input); cudaFree(d_actual);
}