// matmul.cu — Practice 2: 2D Matrix Multiplication & CPU vs GPU Benchmark
//
// TASK FOR YOU:
// Fill in the __global__ kernel function `matmul_kernel` below using what you learned!
//
// Syntax hints:
//  - Calculate 2D global col (x) & row (y):
//      int col = blockIdx.x * blockDim.x + threadIdx.x;
//      int row = blockIdx.y * blockDim.y + threadIdx.y;
//  - Bounds check for 2D matrix: if (row >= M || col >= N) return;
//  - Row-major indexing: A[row * K + k] and B[k * N + col]
//  - Accumulate sum in a loop over k (from 0 to K-1) and write C[row * N + col] = sum;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════
// STEP 1: WRITE YOUR 2D CUDA KERNEL HERE
// ═══════════════════════════════════════════════════════════════════
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // TODO: Calculate 2D column (x) and row (y) indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Add 2D bounds check
    if (row >= M || col >= N) return;

    // TODO: Compute dot product of Row 'row' of A and Column 'col' of B
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // TODO: Write output matrix C[row][col]
    C[row * N + col] = sum;
}


// ═══════════════════════════════════════════════════════════════════
// CPU REFERENCE IMPLEMENTATION (Triple-Nested Loop)
// ═══════════════════════════════════════════════════════════════════
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// HARNESS & BENCHMARK SUITE
// ═══════════════════════════════════════════════════════════════════
int main() {
    // Matrix Dimensions: A (256x512) x B (512x256) -> C (256x256)
    int M = 256, K = 512, N = 256;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    printf("=====================================================\n");
    printf("  CUDA Practice 2: Matrix Multiply Benchmark        \n");
    printf("  A[%dx%d] x B[%dx%d] = C[%dx%d]\n", M, K, K, N, M, N);
    printf("  Total Operations: %.2f Million FLOPs\n", (2.0 * M * N * K) / 1e6);
    printf("=====================================================\n\n");

    // 1. Allocate Host Memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_gpu = (float*)malloc(bytes_C);
    float *h_C_cpu = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(i % 13) * 0.1f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i % 7) * 0.05f;

    // 2. Benchmark CPU Execution
    printf("[CPU] Running matmul on CPU (O(N^3) sequential loops)...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("  --> CPU Execution Time: %.3f ms\n\n", cpu_ms);

    // 3. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // 4. Configure 2D Grid & Block Dimensions
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    printf("[GPU] Launching 2D CUDA Kernel...\n");
    printf("  --> Grid Size:  (%d, %d) blocks\n", gridSize.x, gridSize.y);
    printf("  --> Block Size: (%d, %d) threads (%d per block)\n", blockSize.x, blockSize.y, blockSize.x * blockSize.y);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    // KERNEL LAUNCH
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);
    printf("  --> GPU Kernel Execution Time: %.3f ms\n\n", gpu_ms);

    // 5. Copy Result Back & Verify Correctness
    cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > 1e-2) {  // Floating point accumulation tolerance
            if (errors < 5) {
                printf("  [ERROR] Mismatch at index %d: GPU=%.4f, CPU=%.4f\n", i, h_C_gpu[i], h_C_cpu[i]);
            }
            errors++;
        }
    }

    printf("=====================================================\n");
    if (errors == 0) {
        printf("  VERIFICATION: \033[1;32m[PASS]\033[0m All matrix outputs match!\n");
        printf("  SPEEDUP:      \033[1;36m%.2fx Faster on GPU\033[0m\n", cpu_ms / gpu_ms);
    } else {
        printf("  VERIFICATION: \033[1;31m[FAIL]\033[0m %d errors detected.\n", errors);
    }
    printf("=====================================================\n");

    // Cleanup
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return 0;
}
