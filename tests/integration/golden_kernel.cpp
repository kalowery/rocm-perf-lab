#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void increment(int* data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 16) {
        data[i] += 1;
    }
}

int main() {
    const int N = 16;

    int host[N];
    for (int i = 0; i < N; ++i)
        host[i] = i;

    int* device;
    hipMalloc(&device, N * sizeof(int));
    hipMemcpy(device, host, N * sizeof(int), hipMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(16);

    hipLaunchKernelGGL(increment, grid, block, 0, 0, device);
    hipDeviceSynchronize();

    hipMemcpy(host, device, N * sizeof(int), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        printf("%d ", host[i]);

    printf("\n");

    hipFree(device);

    return 0;
}
