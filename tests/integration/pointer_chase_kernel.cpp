#include <hip/hip_runtime.h>
#include <cstdio>

struct Node {
    int next;
    int value;
};

__global__ void pointer_chase(Node* nodes, int* result) {
    int idx = 0;
    int sum = 0;

    while (idx != -1) {
        sum += nodes[idx].value;
        idx = nodes[idx].next;
    }

    result[0] = sum;
}

int main() {
    const int N = 16;

    Node host_nodes[N];

    for (int i = 0; i < N; ++i) {
        host_nodes[i].value = i;
        host_nodes[i].next = (i == N - 1) ? -1 : i + 1;
    }

    Node* d_nodes;
    int* d_result;
    int host_result = 0;

    hipMalloc(&d_nodes, sizeof(Node) * N);
    hipMalloc(&d_result, sizeof(int));

    hipMemcpy(d_nodes, host_nodes, sizeof(Node) * N, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(pointer_chase, dim3(1), dim3(1), 0, 0, d_nodes, d_result);
    hipDeviceSynchronize();

    hipMemcpy(&host_result, d_result, sizeof(int), hipMemcpyDeviceToHost);

    printf("SUM=%d\n", host_result);

    hipFree(d_nodes);
    hipFree(d_result);

    return 0;
}
