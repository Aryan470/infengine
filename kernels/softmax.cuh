#pragma once
#include <cuda_fp16.h>
#include <cstdint>

void scale_causal_softmax(int seq_len, __half* d_input, __half* d_output, const bool is_row);
__global__ void scale_causal_softmax_kern(int seq_len, __half* d_input, __half* d_output);
__global__ void scale_row_softmax_kern(int seq_len, __half* d_input, __half* d_output);

// Tree-masked softmax for speculative decoding verification
// Input: [NUM_Q_HEADS, N_draft, total_seq_len] where total_seq_len = seq_len + N_draft
// tree_mask: [N_draft, N_draft] — tree_mask[i][j] = 1 iff draft token j is visible to draft token i
// Columns 0..seq_len-1 are always visible, columns seq_len..total_seq_len-1 use tree_mask
void tree_masked_softmax(int N_draft, int seq_len, __half* d_input, __half* d_output, const int8_t* d_tree_mask);