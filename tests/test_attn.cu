#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/attn.cuh"
#include "../kernels/rope.cuh"

TEST(Attention, MatchesPyTorch) {
    std::vector<__half> input = load_tensor("test_data/attn_e2e_input.bin");
    std::vector<__half> W_k = load_tensor("test_data/attn_kproj.bin");
    std::vector<__half> W_q = load_tensor("test_data/attn_qproj.bin");
    std::vector<__half> W_v = load_tensor("test_data/attn_vproj.bin");
    std::vector<__half> W_o = load_tensor("test_data/attn_oproj.bin");
    std::vector<__half> expected = load_tensor("test_data/attn_e2e_output.bin");
    std::vector<__half> actual(expected.size());

    // input/output is [seq_len, hidden_dim]
    // Wk = [n_kv_heads, head_dim, hidden_dim]
    // Wq = [n_q_heads, head_dim, hidden_dim]
    // Wv = [n_kv_heads, head_dim, hidden_dim]
    // Wo = [hidden_dim, hidden_dim]
    const int seq_len = input.size() / InfEngineConfig::HIDDEN_SIZE;
    const int input_size_bytes = InfEngineConfig::HALF_SIZE * input.size();
    const int output_size_bytes = input_size_bytes;

    const int kv_proj_size_bytes = InfEngineConfig::NUM_KV_HEADS * InfEngineConfig::HEAD_DIM * InfEngineConfig::HIDDEN_SIZE * InfEngineConfig::HALF_SIZE;
    const int q_proj_size_bytes = InfEngineConfig::NUM_Q_HEADS * InfEngineConfig::HEAD_DIM * InfEngineConfig::HIDDEN_SIZE * InfEngineConfig::HALF_SIZE;
    const int o_proj_size_bytes = InfEngineConfig::HIDDEN_SIZE * InfEngineConfig::HIDDEN_SIZE * InfEngineConfig::HALF_SIZE;

    __half* d_input; __half* d_Wk; __half* d_Wq; __half* d_Wv; __half* d_Wo; __half* d_actual;
    cudaMalloc(&d_input, input_size_bytes);
    cudaMemcpy(d_input, input.data(), input_size_bytes, cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_Wk, kv_proj_size_bytes);
    cudaMemcpy(d_Wk, W_k.data(), kv_proj_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_Wq, q_proj_size_bytes);
    cudaMemcpy(d_Wq, W_q.data(), q_proj_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_Wv, kv_proj_size_bytes);
    cudaMemcpy(d_Wv, W_v.data(), kv_proj_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_Wo, o_proj_size_bytes);
    cudaMemcpy(d_Wo, W_o.data(), o_proj_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_actual, output_size_bytes);

    // copy the cuda buffer into actual rope cos and actual rope sin
    float* d_rope_cos;
    float* d_rope_sin;
    init_rope_buffer(&d_rope_cos, &d_rope_sin);

    // needs to be launched with 
    attn(d_rope_cos, d_rope_sin, seq_len, d_input, d_Wq, d_Wk, d_Wv, d_Wo, d_actual);
    cudaMemcpy(actual.data(), d_actual, output_size_bytes, cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 1e-3);

    EXPECT_LT(result.max_abs_err, 3e-3);
    EXPECT_GT(result.pct_within_tol, 99.9);

    cudaFree(d_input);
    cudaFree(d_Wk);
    cudaFree(d_Wq);
    cudaFree(d_Wv);
    cudaFree(d_Wo);
    cudaFree(d_actual);
    cleanup_rope_buffer(d_rope_cos, d_rope_sin);
}