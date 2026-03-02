#include <cublas_v2.h>
// implemented in ffn.cu. making it visible to others
void multiply_by_weight(cublasHandle_t handle, const int m, const int k, const int n, const half* d_in, const half* d_w, half* d_out);