#pragma once
#include <cuda_fp16.h>

void sample_token_amax(const half* d_logits, int* d_output);

// Batched argmax: run argmax on each of N rows of [VOCAB_SIZE] logits
// d_logits: [N, VOCAB_SIZE], d_output: [N] with argmax token ID per row
void sample_tokens_amax(const half* d_logits, int* d_output, int N);