#include "../config.h"
#include "softmax.cuh"
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

const int K = 512;

// take in QK^T [num_q_heads, seq_len, seq_len], return [num_q_heads, seq_len, seq_len] (attn scores)
void scale_causal_softmax(int seq_len, __half* d_input, __half* d_output, const bool is_row) {
    // we will assign 1 block for each row
    // num_q_heads * seq_len blocks, each one has k threads
    // shmem size will be some space for max and half reduction, and some more space for storing the data itself
    int shmem_size = K * sizeof(float) + seq_len * sizeof(half);
    if (is_row) {
        scale_row_softmax_kern<<<InfEngineConfig::NUM_Q_HEADS, K, shmem_size>>>(seq_len, d_input, d_output);
    } else {
        scale_causal_softmax_kern<<<InfEngineConfig::NUM_Q_HEADS * seq_len, K, shmem_size>>>(seq_len, d_input, d_output);
    }
}

// in the reduce space with K threads, we will find max into shmem[0]
// TODO: should we template this into op reduce?
__device__ half reduce_max(half my_val, half* reduce_space) {
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    reduce_space[thread_id] = my_val;

    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            reduce_space[thread_id] = __hmax(reduce_space[thread_id], reduce_space[thread_id + delta]);
        }
        __syncthreads();
    }
    return reduce_space[0];
}

__device__ float reduce_sum(float my_val, float* reduce_space) {
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    reduce_space[thread_id] = my_val;

    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            reduce_space[thread_id] += reduce_space[thread_id + delta];
        }
        __syncthreads();
    }
    return reduce_space[0];
}


__global__ void scale_causal_softmax_kern(int seq_len, half* d_input, half* d_output) {
    extern __shared__ float raw_shmem[];
    // where we do reductions (max, sum)
    float* reduce_space = raw_shmem;
    // where we actually store data
    // TODO: this assumes the entire seq len fits in shmem. this is not true for very long context
    half* data_space = (half*) (raw_shmem + K);


    // each block operates on an attention score row for a particular (q_head, token)

    // to cover the full row, i will do element tid, tid + delta, tid + 2delta, ...
    // d_input has many rows, each of len seq_len. mine is at block_id
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    const int row_idx = block_id % seq_len;
    // row len / num threads, rounded up
    const int stride = num_threads;

    half* block_input = d_input + (seq_len * block_id);
    half* block_output = d_output + (seq_len * block_id);

    const float scale_value = rsqrt(1.0f * InfEngineConfig::HEAD_DIM);
    // load in the data to shmem
    half local_max = __float2half(-INFINITY);
    for (int i = thread_id; i < seq_len; i += stride) {
        // if we are on LHS incl diag, load the value. otherwise write -inf
        data_space[i] = __float2half(i <= row_idx ? __half2float(block_input[i]) * scale_value : -INFINITY);
        local_max = __hmax(local_max, data_space[i]);
    }
    __syncthreads();
    // once all the loading is done, we need to find max
    float global_max = __half2float(reduce_max(local_max, (half*) reduce_space));

    // now that we have the global max, we go back and subtract, exponentiate
    float local_sum = 0.0f;
    for (int i = thread_id; i < seq_len; i += stride) {
        // subtract, exponentiate, and sum
        float new_data = exp(__half2float(data_space[i]) - global_max);
        local_sum += new_data;
        data_space[i] = __float2half(new_data);
    }
    __syncthreads();

    float global_sum = reduce_sum(local_sum, reduce_space);
    float inv_global_sum = 1.0f / global_sum;
    // now divide everything by sum and write to output
    for (int i = thread_id; i < seq_len; i += stride) {
        block_output[i] = __float2half(__half2float(data_space[i]) * inv_global_sum);
    }
}

__global__ void tree_masked_softmax_kern(int N_draft, int seq_len, int total_seq_len, half* d_input, half* d_output, const int8_t* d_tree_mask) {
    extern __shared__ float raw_shmem[];
    float* reduce_space = raw_shmem;
    // No data_space in shmem for this kernel — total_seq_len can be large
    // We use register-based multi-pass approach

    // Each block handles one row: block_id = head_id * N_draft + draft_token_idx
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    const int draft_idx = block_id % N_draft;  // which draft token row

    half* row_input = d_input + (size_t)block_id * total_seq_len;
    half* row_output = d_output + (size_t)block_id * total_seq_len;
    const int8_t* my_mask = d_tree_mask + draft_idx * N_draft;  // [N_draft]

    const float scale_value = rsqrt(1.0f * InfEngineConfig::HEAD_DIM);

    // Pass 1: find max
    float local_max = -INFINITY;
    for (int i = thread_id; i < total_seq_len; i += num_threads) {
        float val;
        if (i < seq_len) {
            // Always visible (prefix KV cache)
            val = __half2float(row_input[i]) * scale_value;
        } else {
            int draft_col = i - seq_len;
            if (my_mask[draft_col]) {
                val = __half2float(row_input[i]) * scale_value;
            } else {
                val = -INFINITY;
            }
        }
        if (val > local_max) local_max = val;
    }

    // Reduce max
    reduce_space[thread_id] = local_max;
    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            reduce_space[thread_id] = fmaxf(reduce_space[thread_id], reduce_space[thread_id + delta]);
        }
        __syncthreads();
    }
    float global_max = reduce_space[0];

    // Pass 2: exponentiate and sum
    float local_sum = 0.0f;
    for (int i = thread_id; i < total_seq_len; i += num_threads) {
        float val;
        if (i < seq_len) {
            val = __half2float(row_input[i]) * scale_value;
        } else {
            int draft_col = i - seq_len;
            if (my_mask[draft_col]) {
                val = __half2float(row_input[i]) * scale_value;
            } else {
                val = -INFINITY;
            }
        }
        float exp_val = expf(val - global_max);
        local_sum += exp_val;
    }

    // Reduce sum
    reduce_space[thread_id] = local_sum;
    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            reduce_space[thread_id] += reduce_space[thread_id + delta];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_space[0];

    // Pass 3: normalize and write
    for (int i = thread_id; i < total_seq_len; i += num_threads) {
        float val;
        if (i < seq_len) {
            val = __half2float(row_input[i]) * scale_value;
        } else {
            int draft_col = i - seq_len;
            if (my_mask[draft_col]) {
                val = __half2float(row_input[i]) * scale_value;
            } else {
                val = -INFINITY;
            }
        }
        float exp_val = expf(val - global_max);
        row_output[i] = __float2half(exp_val * inv_sum);
    }
}

void tree_masked_softmax(int N_draft, int seq_len, half* d_input, half* d_output, const int8_t* d_tree_mask) {
    int total_seq_len = seq_len + N_draft;
    int num_blocks = InfEngineConfig::NUM_Q_HEADS * N_draft;
    int shmem_size = K * sizeof(float);
    tree_masked_softmax_kern<<<num_blocks, K, shmem_size>>>(N_draft, seq_len, total_seq_len, d_input, d_output, d_tree_mask);
}

__global__ void scale_row_softmax_kern(int seq_len, half* d_input, half* d_output) {
    extern __shared__ float raw_shmem[];
    // where we do reductions (max, sum)
    float* reduce_space = raw_shmem;
    // where we actually store data
    // TODO: this assumes the entire seq len fits in shmem. this is not true for very long context
    half* data_space = (half*) (raw_shmem + K);


    // each block operates on an attention score row for a particular (q_head, token)

    // to cover the full row, i will do element tid, tid + delta, tid + 2delta, ...
    // d_input has many rows, each of len seq_len. mine is at block_id
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    // row len / num threads, rounded up
    const int stride = num_threads;

    half* block_input = d_input + (seq_len * block_id);
    half* block_output = d_output + (seq_len * block_id);

    const float scale_value = rsqrt(1.0f * InfEngineConfig::HEAD_DIM);
    // load in the data to shmem
    half local_max = __float2half(-INFINITY);
    for (int i = thread_id; i < seq_len; i += stride) {
        // if we are on LHS incl diag, load the value. otherwise write -inf
        data_space[i] = __float2half(__half2float(block_input[i]) * scale_value);
        local_max = __hmax(local_max, data_space[i]);
    }
    __syncthreads();
    // once all the loading is done, we need to find max
    float global_max = __half2float(reduce_max(local_max, (half*) reduce_space));

    // now that we have the global max, we go back and subtract, exponentiate
    float local_sum = 0.0f;
    for (int i = thread_id; i < seq_len; i += stride) {
        // subtract, exponentiate, and sum
        float new_data = exp(__half2float(data_space[i]) - global_max);
        local_sum += new_data;
        data_space[i] = __float2half(new_data);
    }
    __syncthreads();

    float global_sum = reduce_sum(local_sum, reduce_space);
    float inv_global_sum = 1.0f / global_sum;
    // now divide everything by sum and write to output
    for (int i = thread_id; i < seq_len; i += stride) {
        block_output[i] = __float2half(__half2float(data_space[i]) * inv_global_sum);
    }
}