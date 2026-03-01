#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <library_types.h>
#include "../config.h"
#include "rope.cuh"
#include "softmax.cuh"
#include "attn_permute.cuh"

void multiply_kqv_proj(cublasHandle_t handle, const half* d_W, const half* d_input, half* d_out, const int seq_len, const int num_heads) {
    float alpha = 1.0f;
    float beta = 0.0f;
    auto proj_result = cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, // transpose W_k
        CUBLAS_OP_N, // not transpose x
        InfEngineConfig::HEAD_DIM, // m = 128
        seq_len, // n = seq_len
        InfEngineConfig::HIDDEN_SIZE, // k = 4096
        &alpha,
        d_W, // A = W_k
        CUDA_R_16F, // A is fp16
        InfEngineConfig::HIDDEN_SIZE, // lda = num rows in A in colmajor view = 4096
        InfEngineConfig::HEAD_DIM * InfEngineConfig::HIDDEN_SIZE, // size of A matrix = m * k
        d_input,
        CUDA_R_16F, // B is fp16
        InfEngineConfig::HIDDEN_SIZE, // this is viewed as (hidden_dim, seq_len)
        0, // do not stride B
        &beta, // betaC = 0 (ignore curr values in C)
        d_out, // output buffer
        CUDA_R_16F, // C is fp16
        InfEngineConfig::HEAD_DIM, // this is viewed as (head_dim, seq_len)
        InfEngineConfig::HEAD_DIM * seq_len, // stride C by seq_len * head_dim
        num_heads, // batchCount = n_kv_heads
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
}

void multiply_QKt(cublasHandle_t handle, const half* Q, const half* K, half* d_QKt_buffer, const int seq_len) {
    // another matmul... we want to find X = Q @ K^t -> X^t = K @ Q^t -> we give cuBLAS K, tell it to transpose, give it Q, tell it not to transpose
    // we want to do n_q_heads matmuls, but those matmuls will share K matrices, we will use a ptr array to write this
    // as we do matmuls, we keep advancing q. we advance k once every (num_q/num_k). we keep advancing out.
    const half* q_ptrs[InfEngineConfig::NUM_Q_HEADS];
    const half* k_ptrs[InfEngineConfig::NUM_Q_HEADS];
    half* o_ptrs[InfEngineConfig::NUM_Q_HEADS];
    q_ptrs[0] = Q;
    k_ptrs[0] = K;
    o_ptrs[0] = d_QKt_buffer;

    for (int i = 1; i < InfEngineConfig::NUM_Q_HEADS; i++) {
        // move up by size of 1 Q matrix = seq_len * head_dim
        q_ptrs[i] = q_ptrs[i - 1] + (seq_len * InfEngineConfig::HEAD_DIM);
        // if i % (num_q/num_k) = 0, then move up by 1 k matrix (seq_len * head_dim)
        k_ptrs[i] = k_ptrs[i - 1];
        if (i % (InfEngineConfig::NUM_Q_HEADS / InfEngineConfig::NUM_KV_HEADS) == 0) {
            k_ptrs[i] += seq_len * InfEngineConfig::HEAD_DIM;
        }
        // move up by size of 1 output matrix = seq_len * seq_len
        o_ptrs[i] = o_ptrs[i - 1] + (seq_len * seq_len);
    }

    float alpha = 1.0f; float beta = 0.0f;

    // we actually have to put the q/k/output ptr arrays on device
    half** d_q_ptrs; half** d_k_ptrs; half** d_o_ptrs;
    cudaMalloc(&d_q_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMalloc(&d_k_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMalloc(&d_o_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMemcpy(d_q_ptrs, q_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_ptrs, k_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_ptrs, o_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);


    // m: rows in K = seq_len, k = cols in K = rows in Q^t = head_dim, n = cols in Q^t = seq_len
    auto result = cublasGemmBatchedEx(handle,
        CUBLAS_OP_T, // do transpose A (K)
        CUBLAS_OP_N, // do not transpose B (Q)
        seq_len, //m = seq_len
        seq_len, //n = seq_len
        InfEngineConfig::HEAD_DIM, // k = head_dim
        &alpha,
        (const void**) d_k_ptrs,
        CUDA_R_16F,
        InfEngineConfig::HEAD_DIM, // NOTE: we store the actual stored stride here because we applied OP T, so head_dim
        (const void**) d_q_ptrs,
        CUDA_R_16F,
        InfEngineConfig::HEAD_DIM, // ldb = rows in Q^t = head_dim
        &beta,
        (void**) d_o_ptrs,
        CUDA_R_16F,
        seq_len, // ldc = rows in output = seq_len
        InfEngineConfig::NUM_Q_HEADS,
        CUDA_R_32F, // accum in fp32
        CUBLAS_GEMM_DEFAULT);

    cudaFree(d_q_ptrs); cudaFree(d_k_ptrs); cudaFree(d_o_ptrs);
    if (result != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmBatchedEx failed in QK^T computation with error code %d\n", result);
    }

}

void multiply_attn_weights_V(cublasHandle_t handle, const int seq_len, const half* attn_weights, const half* V, half* output) {
    // want to compute attn_weights @ V, call it X = A @ V, however V is broadcasted
    // attn_weights is [n_q_heads, seq_len, seq_len], v is [n_kv_heads, seq_len, head_dim]
    // output should be [num_q_heads, seq_len, head_dim]
    // each of n_q_heads matmul is [seq_len, seq_len] x [seq_len, head_dim]
    // we advance A for each matmul, advance V every N_Q/N_KV matmuls, advance X every matmul
    // we will compute X^t = V^t @ A^t, we can feed in V, A and they will be read as transposed
    // dim V^t = [head_dim, seq_len], A^t = [seq_len, seq_len], X^t = [head_dim, seq_len]

    const half* v_ptrs[InfEngineConfig::NUM_Q_HEADS];
    const half* a_ptrs[InfEngineConfig::NUM_Q_HEADS];
    half* o_ptrs[InfEngineConfig::NUM_Q_HEADS];
    v_ptrs[0] = V;
    a_ptrs[0] = attn_weights;
    o_ptrs[0] = output;

    for (int i = 1; i < InfEngineConfig::NUM_Q_HEADS; i++) {
        // if i % (num_q/num_k) = 0, then move up by 1 V matrix (seq_len * head_dim)
        v_ptrs[i] = v_ptrs[i - 1];
        if (i % (InfEngineConfig::NUM_Q_HEADS / InfEngineConfig::NUM_KV_HEADS) == 0) {
            v_ptrs[i] += seq_len * InfEngineConfig::HEAD_DIM;
        }

        // move up by size of 1 A matrix = seq_len * seq_len
        a_ptrs[i] = a_ptrs[i - 1] + (seq_len * seq_len);
        // move up by size of 1 output matrix = seq_len * head_dim
        o_ptrs[i] = o_ptrs[i - 1] + (seq_len * InfEngineConfig::HEAD_DIM);
    }

    float alpha = 1.0f; float beta = 0.0f;

    // we actually have to put the q/k/output ptr arrays on device
    half** d_v_ptrs; half** d_a_ptrs; half** d_o_ptrs;
    cudaMalloc(&d_v_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMalloc(&d_a_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMalloc(&d_o_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*));
    cudaMemcpy(d_v_ptrs, v_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_ptrs, a_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_ptrs, o_ptrs, InfEngineConfig::NUM_Q_HEADS * sizeof(half*), cudaMemcpyHostToDevice);

    // m: rows in V^t = head_dim, k = cols in V^t = rows in A^t = seq_len, n = cols in A^t = seq_len
    auto result = cublasGemmBatchedEx(handle,
        CUBLAS_OP_N, // don't transpose
        CUBLAS_OP_N, // don't transpose
        InfEngineConfig::HEAD_DIM, // m = head_dim
        seq_len, // n = seq_len
        seq_len, // k = seq_len
        &alpha,
        (const void**) d_v_ptrs,
        CUDA_R_16F,
        InfEngineConfig::HEAD_DIM, // lda= rows in V^t = head_dim
        (const void**) d_a_ptrs,
        CUDA_R_16F,
        seq_len, // ldb = rows in A^t = seq_len
        &beta,
        (void**) d_o_ptrs,
        CUDA_R_16F,
        InfEngineConfig::HEAD_DIM, // ldc = rows in output^t = head_dim
        InfEngineConfig::NUM_Q_HEADS,
        CUDA_R_32F, // accum in fp32
        CUBLAS_GEMM_DEFAULT);

    cudaFree(d_v_ptrs); cudaFree(d_a_ptrs); cudaFree(d_o_ptrs);

    if (result != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmBatchedEx failed in attn_weights @ V computation with error code %d\n", result);
    }
}

void multiply_preproj_Wo(cublasHandle_t handle, const int seq_len, const half* d_input, const half* d_Wo, half* d_output) {
    // num_q_heads * head_dim = hidden_dim
    // simple matmul, A is [seq_len, hidden_dim], B is [hidden_dim, hidden_dim], C = AB
    // so we will compute C^t = B^t A^t. however, W_o is already stored transposed, so we need to retranspose it
    float alpha = 1.0f; float beta = 0.0f;
    auto result = cublasGemmEx(handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        InfEngineConfig::HIDDEN_SIZE, // m = rows in B^t = hidden dim
        seq_len, // n = cols in A^t = seq_len
        InfEngineConfig::HIDDEN_SIZE, // k = cols in B^t = rows in A^t = hidden_dim
        &alpha,
        d_Wo,
        CUDA_R_16F,
        InfEngineConfig::HIDDEN_SIZE, // rows in B^t = hidden_dim
        d_input,
        CUDA_R_16F,
        InfEngineConfig::HIDDEN_SIZE, // rows in A^t = hidden_dim
        &beta,
        d_output,
        CUDA_R_16F,
        InfEngineConfig::HIDDEN_SIZE, // rows in C^t = hidden_dim
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);

    if (result != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmx failed in buffer @ Wo computation with error code %d\n", result);
    }
}

// TODO: prealloc workspace reusable across attn calls
void attn(const float* d_rope_cos, const float* d_rope_sin, const int seq_len, half* d_input, half* d_qproj, half* d_kproj, half* d_vproj, half* d_oproj, half* d_output) {
    // input: [seq_len, hidden_dim]
    // compute K Q V, alloc new buffers for them
    half* K; half* Q; half* V;
    cudaMalloc(&K, seq_len * InfEngineConfig::NUM_KV_HEADS * InfEngineConfig::HEAD_DIM * sizeof(half));
    cudaMalloc(&Q, seq_len * InfEngineConfig::NUM_Q_HEADS * InfEngineConfig::HEAD_DIM * sizeof(half));
    cudaMalloc(&V, seq_len * InfEngineConfig::NUM_KV_HEADS * InfEngineConfig::HEAD_DIM * sizeof(half));

    // input: [seq_len, hidden_dim]
    // W_k: [n_k_heads * head_dim, hidden_dim]
    // want to actually compute K = input @ W_k^t
    // using cuBLAS (colmajor), we can find K^t = W_k @ input^t
    // we can read K^t to get K going from col to row. cuBLAS will read W_k and input as W_k^t and input^t, so we need a transpose flag on W_k
    // for cuBLAS, m = rows in W_k = 128 (head_dim), k = cols in W_k = rows in x = 4096 (hidden_dim), n = cols in x^t (seq_len)
    cublasHandle_t handle;
    cublasCreate(&handle);
    multiply_kqv_proj(handle, d_qproj, d_input, Q, seq_len, InfEngineConfig::NUM_Q_HEADS);
    multiply_kqv_proj(handle, d_kproj, d_input, K, seq_len, InfEngineConfig::NUM_KV_HEADS);
    multiply_kqv_proj(handle, d_vproj, d_input, V, seq_len, InfEngineConfig::NUM_KV_HEADS);

    // now K,Q,V = [n_heads, seq_len, head_dim]
    // apply rope in place to Q,K (d_in = d_out)
    apply_rope(InfEngineConfig::NUM_KV_HEADS, seq_len, d_rope_cos, d_rope_sin, K, K);
    apply_rope(InfEngineConfig::NUM_Q_HEADS, seq_len, d_rope_cos, d_rope_sin, Q, Q);

    // broadcast and transpose done by cublas to get QK^t scores in new buffer [n_heads, seq, seq]
    half* d_attn_weights;
    cudaMalloc(&d_attn_weights, InfEngineConfig::NUM_Q_HEADS * seq_len * seq_len * sizeof(half));
    // [n_q_heads, seq_len, head_dim], [(broadcast)n_kv_heads, seq_len, head_dim] -> [n_q_heads, seq_len, seq_len]
    multiply_QKt(handle, Q, K, d_attn_weights, seq_len);

    // apply in place scale+causal+softmax to get attn_weights in same [n_heads, seq, seq] buffer
    scale_causal_softmax(seq_len, d_attn_weights, d_attn_weights);

    // do batched (across n_heads) attn_weights @ V into the reused Q buffer
    half* d_attended = Q;
    // [n_q_heads, seq_len, seq_len] @ [n_kv_heads (broadcast), seq_len, head_dim] -> [n_q_heads, seq_len, head_dim]
    multiply_attn_weights_V(handle, seq_len, d_attn_weights, V, d_attended);

    half* d_pre_proj;
    cudaMalloc(&d_pre_proj, seq_len * InfEngineConfig::NUM_Q_HEADS * InfEngineConfig::HEAD_DIM * sizeof(half));
    // apply same attn_permute to get to a new buffer [seq_len, num_heads, head_dim]
    attn_permute(seq_len, d_attended, d_pre_proj);

    // buffer @ W_o -> [seq_len, num_heads * head_dim=hidden_dim] @ [hidden_dim, hidden_dim] into output buffer [seq_len, hidden_dim]
    // simple matmul
    multiply_preproj_Wo(handle, seq_len, d_pre_proj, d_oproj, d_output);

    cudaFree(K);
    cudaFree(Q);
    cudaFree(V);
    cudaFree(d_attn_weights);
    cudaFree(d_pre_proj);
    cublasDestroy(handle);
}