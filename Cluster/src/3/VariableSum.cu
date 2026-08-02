#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define threadsPerBlock 256
#define SIZE 1000000

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void computeSum(unsigned char *vector, unsigned long long *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < SIZE) {
        atomicAdd(sum, (unsigned long long)(vector[tid] * vector[tid]));
    }
}

int main(void) {
    unsigned char *d_vector;
    unsigned long long *d_sum;
    cudaError_t err;

    err = cudaMalloc((void **)&d_vector, SIZE * sizeof(unsigned char));
    cudaCheckError(err);

    err = cudaMalloc((void **)&d_sum, sizeof(unsigned long long));
    cudaCheckError(err);

    cudaMemset(d_sum, 0, sizeof(unsigned long long));
    cudaMemset(d_vector, 200, SIZE * sizeof(unsigned char));

    dim3 blockSize(threadsPerBlock);
    dim3 blockCount((SIZE + threadsPerBlock - 1) / threadsPerBlock);

    computeSum<<<blockCount, blockSize>>>(d_vector, d_sum);
    err = cudaGetLastError();
    cudaCheckError(err);

    unsigned long long sum;
    err = cudaMemcpy(&sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaCheckError(err);

    printf("Sum of squares of the vector is %llu\n", sum);

    cudaFree(d_vector);
    cudaFree(d_sum);

    return 0;
}
