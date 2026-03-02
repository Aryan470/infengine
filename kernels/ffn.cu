#include <cublas_v2.h>
#include "../config.h"
#include "ffn.cuh"
#include "swiglu.cuh"

// computes: y = x @ W^t with cuBLAS. pass in features of x, W
void multiply_by_weight(cublasHandle_t handle, const int m, const int k, const int n, const half* d_in, const half* d_w, half* d_out) {
    // to compute Y = X @ W^t, we can find Y^t = W @ X^t. cuBLAS will read X as X^t, but needs to transpose W^t into W. we read back Y^t as Y
    // m = rows in W = input n
    // k = regular k
    // n = cols in X^t = rows in X = input m
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto result = cublasGemmEx(handle,
        CUBLAS_OP_T, // do transpose W^t into W
        CUBLAS_OP_N, // don't transpose X^t
        n, // input n -> m
        m, // input m -> n
        k, // regular k
        &alpha,
        d_w,
        CUDA_R_16F,
        k, // space between rows in W is k
        d_in,
        CUDA_R_16F,
        k, // space between rows in X^t is k
        &beta,
        d_out,
        CUDA_R_16F,
        n, // space between rows in output = num output cols = n
        CUDA_R_32F, // do compute in fp32
        CUBLAS_GEMM_DEFAULT);
}

size_t ffn_workspace_size(int max_seq_len) {
    return 2 * (size_t)max_seq_len * InfEngineConfig::FFN_DIM * sizeof(half);
}

void ffn(cublasHandle_t handle, const int seq_len, half* d_in, half* d_out, half* d_wup, half* d_wdown, half* d_wgate, void* workspace) {
    // x = input [seq_len, hidden_dim]
    // Wg is [ffn_dim, hidden_dim]
    // Wup is [ffn_dim, hidden_dim]
    // Wdown is [hidden_dim, ffn_dim]

    // G = x @ Wg^t [seq_len, ffn_dim]
    // U = x @ Wu^t [seq_len, ffn_dim]

    // unpack workspace into 2 buffers for [seq_len, ffn_dim]
    half* d_G = (half*)workspace;
    half* d_U = d_G + seq_len * InfEngineConfig::FFN_DIM;

    // x = [seq_len, hidden_dim], W = [ffn_dim, hidden_dim] -> m = seq_len, k = hidden_dim, n = ffn_dim
    multiply_by_weight(handle, seq_len, InfEngineConfig::HIDDEN_SIZE, InfEngineConfig::FFN_DIM, d_in, d_wup, d_U);
    multiply_by_weight(handle, seq_len, InfEngineConfig::HIDDEN_SIZE, InfEngineConfig::FFN_DIM, d_in, d_wgate, d_G);

    // reuse G buffer for SwiGLU: G' = SwiGLU(G, U)
    swiglu(seq_len, d_G, d_U, d_G);

    // downproj into output:
    // output = G' @ Wo^t -> G = [seq_len, ffn_dim], W_o = [hidden_dim, ffn_dim] -> m = seq_len, k = ffn_dim, n = hidden_dim
    multiply_by_weight(handle, seq_len, InfEngineConfig::FFN_DIM, InfEngineConfig::HIDDEN_SIZE, d_G, d_wdown, d_out);

    // output [seq_len, hidden_dim]
}