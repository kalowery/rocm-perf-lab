#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>

#define N 1024
#define ROWS 1024

__global__ void naive_softmax(float* input, float* output) {
    int row = blockIdx.x;
    if (row >= ROWS) return;

    // First pass: find max
    float max_val = -1e30f;
    for (int i = 0; i < N; ++i) {
        float val = input[row * N + i];
        if (val > max_val) max_val = val;
    }

    // Second pass: compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = expf(input[row * N + i] - max_val);
        output[row * N + i] = e;
        sum += e;
    }

    // Third pass: normalize
    for (int i = 0; i < N; ++i) {
        output[row * N + i] /= sum;
    }
}

int main() {
    float* h_input = new float[ROWS * N];
    float* h_output = new float[ROWS * N];

    for (int i = 0; i < ROWS * N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_input, *d_output;
    hipMalloc(&d_input, ROWS * N * sizeof(float));
    hipMalloc(&d_output, ROWS * N * sizeof(float));

    hipMemcpy(d_input, h_input, ROWS * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid(ROWS);
    dim3 block(1);

    naive_softmax<<<grid, block>>>(d_input, d_output);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, ROWS * N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);

    delete[] h_input;
    delete[] h_output;

    std::cout << "Softmax completed." << std::endl;
    return 0;
}
