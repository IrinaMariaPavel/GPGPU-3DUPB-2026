NVCC ?= nvcc
# A100 Cluster GPU architecture is sm_80. You can check on https://developer.nvidia.com/cuda-gpus.
CUDA_ARCH ?= sm_80
NVCCFLAGS ?= -O2 -arch=$(CUDA_ARCH)

all: countingGFLOPS cudaProperties

countingGFLOPS: src/1/CountingGFLOPS.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cudaProperties: src/2/CudaProperties.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f countingGFLOPS cudaProperties

.PHONY: all clean
