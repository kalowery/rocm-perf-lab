#include <hip/hip_runtime.h>
#include <iostream>

__global__ void saxpy(float a, float* x, float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *x, *y, *out;
    hipMalloc(&x, bytes);
    hipMalloc(&y, bytes);
    hipMalloc(&out, bytes);

    hipMemset(x, 0, bytes);
    hipMemset(y, 0, bytes);

    saxpy<<<(N+255)/256, 256>>>(2.0f, x, y, out, N);
    hipDeviceSynchronize();

    hipFree(x);
    hipFree(y);
    hipFree(out);

    std::cout << "Done\n";
    return 0;
}
