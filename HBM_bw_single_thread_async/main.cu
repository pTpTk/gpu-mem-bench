#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>

__global__ void kernel(unsigned int* d_data,
                         unsigned int* d_latency_out,
                         unsigned int data_size) {
    extern __shared__ uint smem[];
    unsigned int address = 0;

    volatile unsigned long long accumulator = 0;
    volatile unsigned long long value = 0;

    __syncthreads();
    uint* smem_ptr = smem;
    uint* gmem_ptr = d_data;

    auto start = clock64();
#pragma unroll 1
    for (int i = 0; i < data_size; i+=4) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16));
        smem_ptr += 4;
        gmem_ptr += 4;
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    auto end = clock64();
    // Main measurement loop.
#pragma unroll 1
    for (int i = 0; i < data_size; i++) {
         accumulator += smem[i];
         value += accumulator;
    }


    __syncthreads();
    // Dummy write to force retention.
    d_data[0] = value;

    printf("clock diff %ldus\n", (end - start)/3090);

}

int main(int argc, char* argv[]) {
    unsigned int threads_per_CTA = 1;
    unsigned int CTAs_per_SM = 1;
    
    // Query device properties.
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int numSM = devProp.multiProcessorCount;
    int smem_size = devProp.sharedMemPerBlockOptin;
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    // Compute usable data size from L2 cache.
    
    size_t data_size = smem_size / sizeof(unsigned int);
    printf("data size = %d\n", data_size);
    
    // Allocate and initialize data array.
    unsigned int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, data_size * sizeof(unsigned int));
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to allocate device memory for data (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    unsigned int* h_data = (unsigned int*)malloc(data_size * sizeof(unsigned int));
    for (unsigned int i = 0; i < data_size; i++) {
         h_data[i] = 1;
    }
    err = cudaMemcpy(d_data, h_data, data_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to copy data from host to device (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    free(h_data);
    
    // Set grid and block dimensions.
    dim3 grid(numSM * CTAs_per_SM);
    dim3 block(threads_per_CTA);
    
    // Allocate dynamic shared memory (one unsigned int per thread).
    size_t sharedMemSize = threads_per_CTA * sizeof(unsigned int);

    kernel<<<1, 1, smem_size>>>(d_data, NULL, data_size);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to launch kernel (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    
    // Cleanup.
    cudaFree(d_data);

    return 0;
}
