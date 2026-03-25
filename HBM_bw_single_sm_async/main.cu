#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>
#include <nvml.h>

__global__ void kernel(unsigned int* d_data,
                         unsigned int freq,
                         unsigned int data_size) {
    extern __shared__ uint smem[];

    volatile unsigned long long accumulator = 0;
    volatile unsigned long long value = 0;

    uint segment = data_size / 4 / blockDim.z;
    uint offset = ((32 * threadIdx.z + threadIdx.x) / 8) * segment;

    uint* smem_ptr = smem + offset;
    uint* gmem_ptr = d_data + offset;

    __syncthreads();
    // Main measurement loop.

    auto start = clock64();
#pragma unroll 1
    for (int i = 0; i < segment; i+=32) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16));
        smem_ptr += 32;
        gmem_ptr += 32;
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    auto end = clock64();

    if(threadIdx.x == 0) {
        #pragma unroll 1
        for (int i = 0; i < data_size; i++) {
            accumulator += smem[i];
            value += accumulator;
        }

        // Dummy write to force retention.
        d_data[threadIdx.x] = value;
        printf("clock diff %.2fus\n", (float)(end - start)/freq);
    }

}

int main(int argc, char* argv[]) {
    int z = atoi(argv[1]);

    // Query device properties.
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int smem_size = devProp.sharedMemPerBlockOptin;
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    
    smem_size /= (512*z);
    smem_size *= (512*z);

    // Compute usable data size from L2 cache.
    
    size_t data_size = smem_size / sizeof(unsigned int);
    printf("data size = %ld\n", data_size);
    
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

    nvmlInit();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);
    unsigned int sm_clock;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clock);

    printf("Current SM clock: %u MHz\n", sm_clock);

    kernel<<<1, dim3(32, 1, z), smem_size>>>(d_data, sm_clock, data_size);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to launch kernel (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    
    // Cleanup.
    cudaFree(d_data);
    nvmlShutdown();

    return 0;
}
