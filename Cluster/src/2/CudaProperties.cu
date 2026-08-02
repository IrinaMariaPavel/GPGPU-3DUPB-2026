#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

int _ConvertSMVer2Cores(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x30: return 192;
        case 0x32: return 192;
        case 0x35: return 192;
        case 0x37: return 192;
        case 0x50: return 128;
        case 0x52: return 128;
        case 0x53: return 128;
        case 0x60: return 64;
        case 0x61: return 128;
        case 0x62: return 128;
        case 0x70: return 64;
        case 0x72: return 64;
        case 0x75: return 64;
        case 0x80: return 64;
        case 0x86: return 128;
        case 0x87: return 128;
        case 0x89: return 128;
        case 0x90: return 128;
        default: return 64;
    }
}

int main() {
    int nDevices;
    cudaError_t err;

    err = cudaGetDeviceCount(&nDevices);
    cudaCheckError(err);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        cudaCheckError(err);

        printf("Device name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));

        int multiProcessorCount = prop.multiProcessorCount;
        int coresPerSM = _ConvertSMVer2Cores(prop.major, prop.minor);
        int SPcores = multiProcessorCount * coresPerSM;

        printf("Number of SMs: %d\n", multiProcessorCount);
        printf("Number of SP cores: %d\n\n", SPcores);
    }

    return 0;
}
