#include <driver_types.h>
#include <vector>
#include <cmath>
#include <numbers>
#include "../config.h"
#include "rope.cuh"

const float THETA = 500000.0f;
const int CTX_FACTOR = 8;
const int LOW_FREQ_FACTOR = 1;
const int HIGH_FREQ_FACTOR = 4;
const int ORIGINAL_MAX_POS = 8192;

const int LOW_FREQ_WAVELEN  = ORIGINAL_MAX_POS / LOW_FREQ_FACTOR;
const int HIGH_FREQ_WAVELEN = ORIGINAL_MAX_POS / HIGH_FREQ_FACTOR;

// K should not exceed HEAD_DIM/2
const int K = InfEngineConfig::HEAD_DIM/2;

void apply_rope(int num_heads, int seq_len, const float* d_rope_cos, const float* d_rope_sin, __half* d_input, __half* d_output) {
    apply_rope_kern<<<num_heads * seq_len, K>>>(seq_len, d_rope_cos, d_rope_sin, d_input, d_output);
}

/*
    inputs: cos table [max_seq_len, head_dim/2] (sin is same), d_input [num_heads, seq_len, head_dim]
    outputs: d_output [num_heads, seq_len, head_dim]
    one block per token / seq_len, K threads responsible for pairs of the head dim
*/
__global__ void apply_rope_kern(int seq_len, const float* d_rope_cos, const float* d_rope_sin, half* d_input, half* d_output) {
    const int block_id = blockIdx.x;
    const int threads_per_block = blockDim.x;
    const int thread_id = threadIdx.x;
    const int seq_offset = block_id % seq_len;

    // my block is responsible for 128 (HEAD_DIM) entries at d_input[seq_len=block_id][:]
    // i am responsible for (128/2 pairs) / threads_per_block pairs
    const int my_num_pairs = (InfEngineConfig::HEAD_DIM / 2) / threads_per_block;
    half* block_input = d_input + block_id * InfEngineConfig::HEAD_DIM + thread_id * my_num_pairs;
    half* block_output = d_output + block_id * InfEngineConfig::HEAD_DIM + thread_id * my_num_pairs;

    // [seq_len, head_dim/2]
    const float* my_cos = d_rope_cos + seq_offset * InfEngineConfig::HEAD_DIM / 2 + thread_id * my_num_pairs;
    const float* my_sin = d_rope_sin + seq_offset * InfEngineConfig::HEAD_DIM / 2 + thread_id * my_num_pairs;

    for (int i = 0; i < my_num_pairs; i++) {
        /*
          let a = x[2i], b = x[2i+1]
            a' = a * cos[pos][i] - b * sin[pos][i]
            b' = a * sin[pos][i] + b * cos[pos][i]
        */
        const int x = 2 * i;
        const int y = x + InfEngineConfig::HEAD_DIM / 2;
        float a = __half2float(block_input[x]);
        float b = __half2float(block_input[y]);
        block_output[x] = __float2half(a * my_cos[i] - b * my_sin[i]);
        block_output[y] = __float2half(a * my_sin[i] + b * my_cos[i]);
    }
}

void init_rope_buffer(float** cu_rope_cos, float** cu_rope_sin) {
    // each table is [max_seq_len, headdim/2]
    const int table_size = InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HEAD_DIM / 2;
    std::vector<float> freqs(InfEngineConfig::HEAD_DIM / 2);
    for (int i = 0; i < InfEngineConfig::HEAD_DIM/2; i++) {
        // freq = 1 / (theta ^ (2i / headdim))
        // wavelength = 2pi / freq
        float freq = 1 / std::pow(THETA, 2.0 * i / InfEngineConfig::HEAD_DIM);
        float wavelength = 2 * std::numbers::pi / freq;
        if (wavelength > LOW_FREQ_WAVELEN) {
            freqs[i] = freq / CTX_FACTOR;
        } else if (wavelength < HIGH_FREQ_WAVELEN) {
            freqs[i] = freq;
        } else {
            float smoothed = (ORIGINAL_MAX_POS / wavelength - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR);
            freqs[i] = (1 - smoothed) * (freq / CTX_FACTOR) + smoothed * freq;
        }
    }

    std::vector<float> rope_cos(table_size);
    std::vector<float> rope_sin(table_size);
    for (int pos = 0; pos < InfEngineConfig::MAX_CONTEXT_LENGTH; pos++) {
        for (int i = 0; i < InfEngineConfig::HEAD_DIM / 2; i++) {
            float angle = pos * freqs[i];
            int table_pos = pos * InfEngineConfig::HEAD_DIM / 2 + i;
            rope_cos[table_pos] = std::cos(angle);
            rope_sin[table_pos] = std::sin(angle);
        }
    }

    cudaMalloc(cu_rope_cos, sizeof(float) * table_size);
    cudaMalloc(cu_rope_sin, sizeof(float) * table_size);

    cudaMemcpy(*cu_rope_cos, rope_cos.data(), sizeof(float) * table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*cu_rope_sin, rope_sin.data(), sizeof(float) * table_size, cudaMemcpyHostToDevice);
}

void cleanup_rope_buffer(float* cu_rope_cos, float* cu_rope_sin) {
    cudaFree(cu_rope_cos);
    cudaFree(cu_rope_sin);
}