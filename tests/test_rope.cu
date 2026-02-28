#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/rope.cuh"

TEST(RoPE, MatchesPyTorch) {
    // [num_heads, seq_len, head_dim]
    std::vector<__half> q_input = load_tensor("test_data/rope_q_input.bin");
    std::vector<__half> k_input = load_tensor("test_data/rope_k_input.bin");
    std::vector<float> cos_expected = load_float_tensor("test_data/rope_cos.bin");
    std::vector<float> sin_expected = load_float_tensor("test_data/rope_sin.bin");
    std::vector<__half> q_output = load_tensor("test_data/rope_q_output.bin");
    std::vector<__half> k_output = load_tensor("test_data/rope_k_output.bin");

    std::vector<__half> actual_q_output(q_output.size());
    std::vector<__half> actual_k_output(k_output.size());

    const int seq_len = q_input.size() / (InfEngineConfig::NUM_Q_HEADS * InfEngineConfig::HEAD_DIM);

    // copy the cuda buffer into actual rope cos and actual rope sin
    float* d_rope_cos;
    float* d_rope_sin;
    init_rope_buffer(&d_rope_cos, &d_rope_sin);

    // first diff actual cos/expected cos
    std::vector<float> cos_actual(cos_expected.size());
    std::vector<float> sin_actual(sin_expected.size());
    cudaMemcpy(cos_actual.data(), d_rope_cos, sizeof(float) * cos_expected.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sin_actual.data(), d_rope_sin, sizeof(float) * sin_expected.size(), cudaMemcpyDeviceToHost);

    // alloc and copy q,k input output
    __half* d_q_input;
    __half* d_k_input;
    __half* d_q_output;
    __half* d_k_output;
    cudaMalloc(&d_q_input, InfEngineConfig::HALF_SIZE * q_input.size());
    cudaMalloc(&d_k_input, InfEngineConfig::HALF_SIZE * k_input.size());
    cudaMalloc(&d_q_output, InfEngineConfig::HALF_SIZE * q_output.size());
    cudaMalloc(&d_k_output, InfEngineConfig::HALF_SIZE * k_output.size());

    cudaMemcpy(d_k_input, k_input.data(), InfEngineConfig::HALF_SIZE * k_input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_input, q_input.data(), InfEngineConfig::HALF_SIZE * q_input.size(), cudaMemcpyHostToDevice);

    apply_rope(InfEngineConfig::NUM_Q_HEADS,  seq_len, d_rope_cos, d_rope_sin, d_q_input, d_q_output);
    apply_rope(InfEngineConfig::NUM_KV_HEADS, seq_len, d_rope_cos, d_rope_sin, d_k_input, d_k_output);

    cudaMemcpy(actual_q_output.data(), d_q_output, InfEngineConfig::HALF_SIZE * q_output.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_k_output.data(), d_k_output, InfEngineConfig::HALF_SIZE * k_output.size(), cudaMemcpyDeviceToHost);

    CompareResult cos_result = compare_float_tensors(cos_actual.data(), cos_expected.data(), cos_expected.size());
    CompareResult sin_result = compare_float_tensors(sin_actual.data(), sin_expected.data(), sin_expected.size());
    CompareResult q_result = compare_tensors(actual_q_output.data(), q_output.data(), q_output.size());
    CompareResult k_result = compare_tensors(actual_k_output.data(), k_output.data(), k_output.size());

    EXPECT_LT(cos_result.max_abs_err, 1e-3);
    EXPECT_GT(cos_result.pct_within_tol, 99.0);
    EXPECT_LT(sin_result.max_abs_err, 1e-3);
    EXPECT_GT(sin_result.pct_within_tol, 99.0);
    EXPECT_LT(q_result.max_abs_err, 2e-3);
    EXPECT_GT(q_result.pct_within_tol, 99.0);
    EXPECT_LT(k_result.max_abs_err, 1e-3);
    EXPECT_GT(k_result.pct_within_tol, 99.0);

    cudaFree(d_k_input); cudaFree(d_q_input);
    cudaFree(d_k_output); cudaFree(d_q_output);
    cleanup_rope_buffer(d_rope_cos, d_rope_sin);
}