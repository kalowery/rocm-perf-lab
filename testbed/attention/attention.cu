#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 1024
#define D 64

__global__ void qk_matmul(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          float* __restrict__ scores) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    for (int col = 0; col < N; ++col) {
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) {
            acc += Q[row * D + k] * K[col * D + k];
        }
        scores[row * N + col] = acc;
    }
}

__global__ void row_softmax(float* __restrict__ scores) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float max_val = -1e30f;
    for (int col = 0; col < N; ++col) {
        float v = scores[row * N + col];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        float e = expf(scores[row * N + col] - max_val);
        scores[row * N + col] = e;
        sum += e;
    }

    for (int col = 0; col < N; ++col) {
        scores[row * N + col] /= sum;
    }
}

__global__ void attn_v_matmul(const float* __restrict__ scores,
                              const float* __restrict__ V,
                              float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    for (int col = 0; col < D; ++col) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += scores[row * N + k] * V[k * D + col];
        }
        output[row * D + col] = acc;
    }
}

int main(int argc, char** argv) {
    int launch_count = 100;
    if (argc > 1) launch_count = atoi(argv[1]);

    size_t qk_size = N * D * sizeof(float);
    size_t score_size = N * N * sizeof(float);
    size_t out_size = N * D * sizeof(float);

    float* h_Q = (float*)malloc(qk_size);
    float* h_K = (float*)malloc(qk_size);
    float* h_V = (float*)malloc(qk_size);

    for (int i = 0; i < N * D; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_scores, *d_out;
    hipMalloc(&d_Q, qk_size);
    hipMalloc(&d_K, qk_size);
    hipMalloc(&d_V, qk_size);
    hipMalloc(&d_scores, score_size);
    hipMalloc(&d_out, out_size);

    hipMemcpy(d_Q, h_Q, qk_size, hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K, qk_size, hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V, qk_size, hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    for (int i = 0; i < launch_count; ++i) {
        qk_matmul<<<grid, block>>>(d_Q, d_K, d_scores);
        row_softmax<<<grid, block>>>(d_scores);
        attn_v_matmul<<<grid, block>>>(d_scores, d_V, d_out);
    }

    hipDeviceSynchronize();

    printf("Attention completed (launch_count=%d).\n", launch_count);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_scores);
    hipFree(d_out);
    free(h_Q);
    free(h_K);
    free(h_V);

    return 0;
}
