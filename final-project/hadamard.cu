
#include "load_ckpt.h"
#include <cnpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda.h>

// Utility: Get next power of 2
int next_power_of_2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Create Hadamard Matrix on Device
void create_hadamard_matrix(int n, float** d_H) {
    std::vector<float> H_host(n * n, 1.0f);
    for (int i = 1; i < n; i <<= 1) {
        for (int y = 0; y < i; ++y) {
            for (int x = 0; x < i; ++x) {
                H_host[(y + i) * n + x] = H_host[y * n + x];
                H_host[y * n + (x + i)] = H_host[y * n + x];
                H_host[(y + i) * n + (x + i)] = -H_host[y * n + x];
            }
        }
    }
    cudaMalloc(d_H, n * n * sizeof(float));
    cudaMemcpy(*d_H, H_host.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
}

// CUDA Matrix Multiplication Kernel with clock64 timing
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K, unsigned long long* d_cycles) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        unsigned long long start = clock64();
    
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    
        unsigned long long end = clock64();
    
        // Only thread (0,0) of (0,0) block stores timing if it's a valid thread
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            d_cycles[0] = end - start;
    }    
}

// Calculate deltaW = H * C * H^T
void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col, int rows, int cols, float** d_deltaW) {
    float* d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_H_row, d_C, d_temp, rows, cols, rows, nullptr);
    cudaDeviceSynchronize();

    cudaMalloc(d_deltaW, rows * cols * sizeof(float));
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_H_col, *d_deltaW, rows, cols, cols, nullptr);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <ct_mat_rows> <ct_mat_cols> <num_tokens> <threads_per_block>"
                  << " <ct_directory> <locs_directory> <x_directory>\n";
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ct_mat_rows_input = std::atoi(argv[1]);
    int ct_mat_cols_input = std::atoi(argv[2]);
    int num_tokens = std::atoi(argv[3]);
    int threads_per_block = std::atoi(argv[4]);
    std::string ct_directory = argv[5];
    std::string locs_directory = argv[6];
    std::string x_directory = argv[7];

    int ct_mat_rows = next_power_of_2(ct_mat_rows_input);
    int ct_mat_cols = next_power_of_2(ct_mat_cols_input);

    float* ct = nullptr;
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    int* locs = nullptr;
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

    float* x = nullptr;
    auto [x_rows, x_cols] = load_ckpt_float(x_directory, x);

    float* ct_mat = new float[ct_mat_rows * ct_mat_cols]();
    for (size_t i = 0; i < locs_cols; ++i) {
        int row = locs[i];
        int col = locs[locs_cols + i];
        ct_mat[row * ct_mat_cols + col] = ct[i];
    }
    std::cout << "\n C matrix loaded\n";

    float* d_C = nullptr;
    cudaMalloc(&d_C, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMemcpy(d_C, ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "\n C matrix copied into device\n";

    float* d_H_row = nullptr;
    float* d_H_col = nullptr;
    create_hadamard_matrix(ct_mat_rows, &d_H_row);
    create_hadamard_matrix(ct_mat_cols, &d_H_col);
    std::cout << "\n Hadamard matrix created\n";

    float* d_x = nullptr;
    cudaMalloc(&d_x, x_rows * x_cols * sizeof(float));
    cudaMemcpy(d_x, x, x_rows * x_cols * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "\n Input X copied into device\n";

    float* d_y = nullptr;
    cudaMalloc(&d_y, x_rows * ct_mat_cols * sizeof(float));

    cudaEventRecord(start);

    float* d_deltaW = nullptr;
    calculate_deltaW(d_H_row, d_C, d_H_col, ct_mat_rows, ct_mat_cols, &d_deltaW);
    std::cout << "\n Delta W completed\n";
    cudaDeviceSynchronize();

    int tx = std::sqrt(threads_per_block);
    int ty = threads_per_block / tx;
    int grid_x = (x_cols + tx - 1) / tx;
    int grid_y = (ct_mat_rows + ty - 1) / ty;
    dim3 threadsPerBlockDim(tx, ty);
    dim3 blocksPerGrid(grid_x, grid_y);

    std::cout << "\n Launching matmul_kernel with: \n";
    std::cout << "\tThreads per block: (" << tx << ", " << ty << ")\n";
    std::cout << "\tBlocks per grid: (" << grid_x << ", " << grid_y << ")\n";

    unsigned long long* d_cycles;
    cudaMalloc(&d_cycles, sizeof(unsigned long long));

    matmul_kernel<<<blocksPerGrid, threadsPerBlockDim>>>(d_x, d_deltaW, d_y, x_rows, ct_mat_cols, x_cols, d_cycles);
    cudaDeviceSynchronize();

    unsigned long long h_cycles = 0;
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    std::cout << "GPU clock cycles used by matmul_kernel: " << h_cycles << "\n";

    cudaFree(d_cycles);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "\n All computations done on CUDA!\n";
    std::cout << "CUDA Execution Time: " << milliseconds << " ms\n";

    cudaFree(d_H_row);
    cudaFree(d_H_col);
    cudaFree(d_C);
    cudaFree(d_deltaW);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] ct_mat;
    delete[] ct;
    delete[] locs;
    delete[] x;

    return 0;
}
