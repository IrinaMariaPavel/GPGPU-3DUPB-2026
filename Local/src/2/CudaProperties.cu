/* Start program to get the number of CUDA cappable devices
   For those devices, get relevant information as:
    - Device name
    - Compute capability
    - Total global memory
    - Number of SMs(SM = Streaming Multiprocessor)
    - Number of SP cores(SP = Stream Processor)
*/
#include <stdio.h>

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

int main() {
    int nDevices;
    cudaError_t err;

    err = cudaGetDeviceCount(&nDevices);
    cudaCheckError(err);

    // The output should be 1 since the only capable one is our main GPU
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Major and minor version of compute capability
    // Those ones can be used to determine the number of cores per SM
    // and the number of SMs per GPU
    int multiProcessorCount = prop.multiProcessorCount;
    // 64 is because prop.major is 7 and prop.minor is 5
    int SPcores = multiProcessorCount * 64;
    printf("Number of SMs: %d\n", multiProcessorCount);
    printf("Number of SP cores: %d\n", SPcores);

}