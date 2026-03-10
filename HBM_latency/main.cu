#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>
#include <nvml.h>

__global__ void pointer_chase_kernel(unsigned int* d_data,
                                     unsigned int freq,
                                     unsigned int iterations,
                                     unsigned int start_index) {
    extern __shared__ unsigned int smem[];

    volatile unsigned long long start = 0;
    volatile unsigned long long end   = 0;
    volatile unsigned int idx = start_index;
    volatile unsigned long long sink = 0;

    __syncthreads();

    // Warm-up: touch a few elements to bring TLB / caches into some state
    idx = d_data[idx];

    __syncthreads();

    start = clock64();
#pragma unroll 1
    for (unsigned int i = 0; i < iterations; ++i) {
        idx = d_data[idx];   // true pointer chasing, each load depends on previous
    }
    end = clock64();

    __syncthreads();

    // Dummy write to force retention and side-effect
    d_data[0] = (unsigned int)idx;

    unsigned long long diff = end - start;
    double avg_cycles = (double)diff / (double)iterations;
    double avg_ns = (avg_cycles / (double)freq) * 1000.0; // freq in MHz, so cycles/freq(MHz) = us

    printf("Total cycles: %llu, iterations: %u, avg cycles: %.2f, avg latency: %.2f ns\n",
           diff, iterations, avg_cycles, avg_ns);
}

int main(int argc, char* argv[]) {
    int device = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    size_t num_elements = 1 << 26; // 16M elements (~64MB if 4B each)
    unsigned int iterations = 1 << 20;
    unsigned int stride = 32; // in elements, controls spatial locality
    unsigned int start_index = 0;

    printf("Device: %s\n", devProp.name);
    printf("num_elements = %zu, iterations = %u, stride = %u\n",
           num_elements, iterations, stride);

    // Allocate and build pointer-chasing structure on host
    unsigned int* h_data = (unsigned int*)malloc(num_elements * sizeof(unsigned int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Build a permutation-like ring with fixed stride
    for (size_t i = 0; i < num_elements; ++i) {
        size_t next = (i + stride) % num_elements;
        h_data[i] = (unsigned int)next;
    }

    // Allocate device memory
    unsigned int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, num_elements * sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy to device
    err = cudaMemcpy(d_data, h_data, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(h_data);

    // Init NVML and query SM clock
    nvmlInit();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(device, &dev);
    unsigned int sm_clock;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clock);

    printf("Current SM clock: %u MHz\n", sm_clock);

    // Launch kernel: single thread, pure latency measurement
    dim3 grid(1);
    dim3 block(1);
    size_t smem_size = 0;

    pointer_chase_kernel<<<grid, block, smem_size>>>(d_data, sm_clock, iterations, start_index);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    cudaFree(d_data);
    nvmlShutdown();

    return 0;
}
