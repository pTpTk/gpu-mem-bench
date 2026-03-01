#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>

//----------------------------------------------------------------------------
// User configurable compile options:
//
//   To choose one memory access mode, compile with one of:
//      -DUSE_STREAM_ACCESS  for stream access
//      -DUSE_STRIDED_ACCESS for strided access
//      -DUSE_RANDOM_ACCESS  for random access mode
//
//   To choose how many steps of random delay to allow, define RANDOM_DELAY_STEPS.
//      (Example: -DRANDOM_DELAY_STEPS=32)
//
//   To enable latency measurement (the clock() calls and latency output), compile with:
//      -DENABLE_LATENCY_MEASUREMENT
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Experiment parameters.
//----------------------------------------------------------------------------

//
// Device function to get SM ID using inline PTX.
//
__device__ unsigned int get_smid(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

//
// Kernel using shared memory to record latency measurements.
// Supports three memory access modes:
//  - In random access mode (-DUSE_RANDOM_ACCESS), the kernel obtains its starting
//    address from a random-address array (d_randAddrs) and updates it every iteration.
//  - In stream access mode (-DUSE_STREAM_ACCESS), the address is computed each iteration.
// Additionally, if ENABLE_RANDOM_DELAY is defined, a delay is inserted.
// Latency measurement (using clock()) is enabled only when ENABLE_LATENCY_MEASUREMENT is defined.
__global__ void kernel(unsigned int* d_data,
                         unsigned int* d_latency_out,
                         unsigned int arr_size) {
    unsigned int smid = get_smid();
    unsigned int address = 0;

    volatile unsigned long long accumulator = 0;
    volatile unsigned long long value = 0;

    __syncthreads();
    auto start = clock64();

    // Main measurement loop.
#pragma unroll 1
    for (int i = 0; i < 1000000; i++) {

         accumulator += d_data[address];
         value += accumulator;
         address += 8;
    }

    auto end = clock64();

    __syncthreads();
    // Dummy write to force retention.
    d_data[address] = value;

    printf("clock diff %ldus\n", (end - start)/3090);

}

//
// Host code (main).
// Command-line options:
//   -t <threads_per_CTA>   (default: 1)
//   -c <CTAs_per_SM>        (default: 1)
//   -o <output_mode>        ('d' for full latency distribution, 'a' for average, 'b' for both)
// The grid dimensions are set based on the number of SMs.
// For full latency distribution, the output file is named as:
//    threads_per_CTA_CTAs_per_SM_accesspattern[optional_rand_delay].log
// For average latency, the file name is based on access pattern and CTAs (without threads_per_CTA)
// and new average values are appended (one per different thread count).
//
int main(int argc, char* argv[]) {
    unsigned int threads_per_CTA = 1;
    unsigned int CTAs_per_SM = 1;
    char output_mode = 'b'; // 'd' = full distribution; 'a' = average; 'b' = both
    
    int opt;
    while ((opt = getopt(argc, argv, "t:c:o:")) != -1) {
        switch(opt) {
            case 't': threads_per_CTA = atoi(optarg); break;
            case 'c': CTAs_per_SM = atoi(optarg); break;
            case 'o': output_mode = optarg[0]; break;
            default:
                printf("Usage: %s -t <THREADS_PER_CTA> -c <CTAs_per_SM> -o <output mode: a, d, or b>\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    // Query device properties.
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int numSM = devProp.multiProcessorCount;
    
    // Compute usable data size from L2 cache.
    size_t total_HBM_bytes = devProp.totalGlobalMem; // in bytes;
    size_t reserved_bytes = 4L * 1024 * 1024 * 1024; // 4GB reserved
    if (total_HBM_bytes <= reserved_bytes) {
        fprintf(stderr, "Error: HBM size (%zu bytes) is insufficient for reserved space (%zu bytes).\n", 
                total_HBM_bytes, reserved_bytes);
        exit(EXIT_FAILURE);
    }
    size_t usable_bytes = total_HBM_bytes - reserved_bytes;
    size_t data_size = 4L * 1024 * 1024 * 1024 / sizeof(unsigned int);
    
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
    

    // Build output filename(s).
    const char* access_pattern = "stream";
    char rand_delay_str[64] = "";

    // Distribution file name: includes threads_per_CTA.
    char dist_filename[64];
    sprintf(dist_filename, "%u_%u_%s%s.log", threads_per_CTA, CTAs_per_SM, access_pattern, rand_delay_str);
    // Average file name: based only on access_pattern and CTAs (aggregated for different thread counts).
    char avg_filename[64];
    sprintf(avg_filename, "%s_%u%s.log", access_pattern, CTAs_per_SM, rand_delay_str);
    
    // Set grid and block dimensions.
    dim3 grid(numSM * CTAs_per_SM);
    dim3 block(threads_per_CTA);
    
    // Allocate dynamic shared memory (one unsigned int per thread).
    size_t sharedMemSize = threads_per_CTA * sizeof(unsigned int);

    kernel<<<1, 1>>>(d_data, NULL, data_size);

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
