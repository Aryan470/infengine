#pragma once
#include <cuda_fp16.h>

void ffn(const int seq_len, half* d_in, half* d_out, half* d_wup, half* d_wdown, half* d_wgate);