#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>

#define N 4096
#define ROWS 8192
#define THREADS_PER_ROW 256

// Naive parallel softmax: still multi-pass, no shared memory,
// no warp shuffles, excessive global memory traffic.
__global__ void naive_softmax(float* input, float* output) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= ROWS) return;

    // 1) Compute max (naive parallel reduction via global memory atomics-like pattern)
    __shared__ float shared_max;
    if (tid == 0) shared_max = -1e30f;
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row * N + i];
        atomicMax((int*)&shared_max, __float_as_int(val));
    }
    __syncthreads();

    float max_val = shared_max;

    // 2) Compute exp and sum (again naive)
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        float e = expf(input[row * N + i] - max_val);
        output[row * N + i] = e;
        atomicAdd(&shared_sum, e);
    }
    __syncthreads();

    float sum = shared_sum;

    // 3) Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[row * N + i] /= sum;
    }
}

int main(int argc, char** argv) {
    size_t total = (size_t)ROWS * N;

    int launch_count = 100; // default
    if (argc > 1) {
        launch_count = atoi(argv[1]);
        if (launch_count <= 0) launch_count = 1;
    }

    float* h_input = new float[total];
    float* h_output = new float[total];

    for (size_t i = 0; i < total; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_input, *d_output;
    hipMalloc(&d_input, total * sizeof(float));
    hipMalloc(&d_output, total * sizeof(float));

    hipMemcpy(d_input, h_input, total * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid(ROWS);
    dim3 block(THREADS_PER_ROW);

    for (int i = 0; i < launch_count; ++i) {
        naive_softmax<<<grid, block>>>(d_input, d_output);
    }
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, total * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);

    delete[] h_input;
    delete[] h_output;

    std::cout << "Softmax completed (launch_count=" << launch_count << ")." << std::endl;
    return 0;
}
