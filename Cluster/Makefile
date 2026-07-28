NVCC ?= nvcc
# TODO: Check the GPU used by the queue which you use on the cluster. (A100 if you use the demo one, dgxa100)
# Modify the CUDA_ARCH variable to match your GPU architecture. You can check your GPU architecture at https://developer.nvidia.com/cuda/gpus and set the appropriate value.
CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -O2 -arch=$(CUDA_ARCH)

all: countingGFLOPS cudaProperties

countingGFLOPS: src/1/CountingGFLOPS.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cudaProperties: src/2/CudaProperties.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

variableSum: src/3/VariableSum.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

matrixMul: src/4/MatrixMultiply.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f countingGFLOPS cudaProperties variableSum matrixMul

.PHONY: all clean
