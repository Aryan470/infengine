#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/ffn.cuh"

TEST(FFN, MatchesPyTorch) {
    std::vector<__half> input = load_tensor("test_data/ffn_in.bin");
    std::vector<__half> expected = load_tensor("test_data/ffn_out.bin");

    std::vector<__half> wgate = load_tensor("test_data/ffn_wgate.bin");
    std::vector<__half> wup = load_tensor("test_data/ffn_wup.bin");
    std::vector<__half> wdown = load_tensor("test_data/ffn_wdown.bin");

    std::vector<__half> actual(expected.size());

    int seq_len = input.size() / InfEngineConfig::HIDDEN_SIZE;

    __half* d_input; __half* d_wgate; __half* d_wup; __half* d_wdown; __half* d_output;

    cudaMalloc(&d_input, InfEngineConfig::HALF_SIZE * input.size());
    cudaMalloc(&d_output, InfEngineConfig::HALF_SIZE * expected.size());
    cudaMalloc(&d_wup, InfEngineConfig::HALF_SIZE * wup.size());
    cudaMalloc(&d_wdown, InfEngineConfig::HALF_SIZE * wdown.size());
    cudaMalloc(&d_wgate, InfEngineConfig::HALF_SIZE * wgate.size());

    cudaMemcpy(d_input, input.data(), InfEngineConfig::HALF_SIZE * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wup, wup.data(), InfEngineConfig::HALF_SIZE * wup.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wdown, wdown.data(), InfEngineConfig::HALF_SIZE * wdown.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wgate, wgate.data(), InfEngineConfig::HALF_SIZE * wgate.size(), cudaMemcpyHostToDevice);

    // needs to be launched with 
    ffn(seq_len, d_input, d_output, d_wup, d_wdown, d_wgate);
    cudaMemcpy(actual.data(), d_output, expected.size() * InfEngineConfig::HALF_SIZE, cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 1e-3);

    EXPECT_LT(result.max_abs_err, 8e-3);
    EXPECT_GT(result.pct_within_tol, 99.0);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_wup); cudaFree(d_wdown); cudaFree(d_wgate);
}