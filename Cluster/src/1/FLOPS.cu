#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define OPS_SCALE (2048)
#define KERNEL_OPS_COUNT (2 * OPS_SCALE)

__global__ void kernel_gflops_fp32(float* a, float* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float x = a[idx];

    #pragma unroll 16
    for (int i = 0; i < OPS_SCALE; i++) {
        x = x * x + x;
    }

    a[idx] = x;
}

__global__ void kernel_gflops_fp64(double* a, double* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double y = (double)a[idx];

    #pragma unroll 16
    for (int i = 0; i < OPS_SCALE; i++) {
        y = y * y + y;
    }

    a[idx] = y;
}

void fill_array_random(float *a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    float *device_a = 0, *device_b = 0;
    float *host_a = 0, *host_b = 0;

    int size = OPS_SCALE * OPS_SCALE;

    host_a = (float *) malloc(size * sizeof(float));
    host_b = (float *) malloc(size * sizeof(float));
    cudaMalloc((void **) &device_a, size * sizeof(float));
    cudaMalloc((void **) &device_b, size * sizeof(float));

    if (host_a == 0 || host_b == 0 || device_a == 0 || device_b == 0) {
        printf("[HOST] Couldn't allocate memory\n");
        return 1;
    }

    cudaError_t err;
    fill_array_random(host_a, size);
    err = cudaMemcpy(device_a, host_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(err);

    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    cudaCheckError(err);
    err = cudaEventCreate(&stop);
    cudaCheckError(err);

    dim3 blockSize(512);
    dim3 blockCount((size + blockSize.x - 1) / blockSize.x);

    err = cudaEventRecord(start, 0);
    cudaCheckError(err);
    
    kernel_gflops_fp32<<<blockCount, blockSize>>>(device_a, device_b);

    err = cudaEventRecord(stop, 0);
    cudaCheckError(err);
    cudaEventSynchronize(stop);

    float ms_fp32 = 0;
    err = cudaEventElapsedTime(&ms_fp32, start, stop);
    cudaCheckError(err);
    float seconds_fp32 = ms_fp32 / 1000.0f;

    double num_ops = (double)KERNEL_OPS_COUNT * size;
    double gflops_fp32 = num_ops / seconds_fp32 / 1e+9;

    printf("FP32 Time: %.5f ms\n", ms_fp32);
    printf("FP32 GFLOPS: %.2f\n\n", gflops_fp32);

    double *device_a_64 = 0, *device_b_64 = 0;
    cudaMalloc((void **) &device_a_64, size * sizeof(double));
    cudaMalloc((void **) &device_b_64, size * sizeof(double));

    err = cudaEventRecord(start, 0);
    cudaCheckError(err);

    kernel_gflops_fp64<<<blockCount, blockSize>>>(device_a_64, device_b_64);

    err = cudaEventRecord(stop, 0);
    cudaCheckError(err);
    cudaEventSynchronize(stop);

    float ms_fp64 = 0;
    err = cudaEventElapsedTime(&ms_fp64, start, stop);
    cudaCheckError(err);
    float seconds_fp64 = ms_fp64 / 1000.0f;

    double gflops_fp64 = num_ops / seconds_fp64 / 1e+9;

    printf("FP64 Time: %.5f ms\n", ms_fp64);
    printf("FP64 GFLOPS: %.2f\n", gflops_fp64);

    free(host_a);
    free(host_b);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_a_64);
    cudaFree(device_b_64);

    return 0;
}
