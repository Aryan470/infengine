#pragma once
#include <cuda_fp16.h>

void residual_add(const int rows, const int cols, half* d_x, half* d_delta);
__global__ void residual_add_kern(const int rows, const int cols, half* d_x, half* d_delta);