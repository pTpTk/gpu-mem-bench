#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>
#include <nvml.h>
#include <cuda/ptx>
#include <cuda/barrier>

#define DATA_SIZE 200000
namespace ptx = cuda::ptx;
using barrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void kernel(unsigned int* d_data,
                         unsigned int freq,
                         unsigned int data_size) {
    extern __shared__ uint smem[];

    volatile unsigned long long accumulator = 0;
    volatile unsigned long long value = 0;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    size_t gmem_addr = static_cast<size_t>(__cvta_generic_to_global(d_data));

    auto start = clock64();

    asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "l"(__cvta_generic_to_shared(smem)), "l"(reinterpret_cast<uint64_t>(d_data)), "n"(232448),
            "l"(__cvta_generic_to_shared(&bar))
        : "memory");

    auto __bh = __cvta_generic_to_shared(cuda::device::barrier_native_handle(bar));
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__bh)),
          "n"(232448)
        : "memory");

    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

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

    printf("clock diff %ldus\n", (end - start)/freq);

}

int main(int argc, char* argv[]) {
    // Query device properties.
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int smem_size = devProp.sharedMemPerBlockOptin;
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

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

    kernel<<<1, 1, smem_size>>>(d_data, sm_clock, data_size);

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
