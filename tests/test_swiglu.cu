#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/swiglu.cuh"

TEST(SwiGLU, MatchesPyTorch) {
    // scalesoftmax takes [q_heads, seq_len, seq_len] -> [q_heads, seq_len, seq_len]
    std::vector<__half> input_gate = load_tensor("test_data/swiglu_gate.bin");
    std::vector<__half> input_up = load_tensor("test_data/swiglu_up.bin");
    std::vector<__half> expected = load_tensor("test_data/swiglu_out.bin");
    std::vector<__half> actual(expected.size());

    const int seq_len = input_gate.size() / InfEngineConfig::FFN_DIM;

    __half* d_gate;
    __half* d_up;
    __half* d_out;

    // size of gate, up, output is all the same
    const int data_size_bytes = seq_len * InfEngineConfig::FFN_DIM * InfEngineConfig::HALF_SIZE;
    cudaMalloc(&d_gate, data_size_bytes);
    cudaMalloc(&d_up, data_size_bytes);
    cudaMalloc(&d_out, data_size_bytes);

    cudaMemcpy(d_gate, input_gate.data(), data_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, input_up.data(), data_size_bytes, cudaMemcpyHostToDevice);

    swiglu(seq_len, d_gate, d_up, d_out);

    cudaMemcpy(actual.data(), d_out, data_size_bytes, cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 2e-5);

    EXPECT_LT(result.max_abs_err, 1e-3);
    EXPECT_GT(result.pct_within_tol, 99.0);

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
}