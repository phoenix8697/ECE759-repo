#include <cnpy.h>
#include "load_ckpt.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

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

// 1D CUDA Matrix Multiplication Kernel
__global__ void matmul_kernel_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// Calculate deltaW = H * C * H^T using 1D kernel
void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col, int rows, int cols, float** d_deltaW, int threads_per_block) {
    float* d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(float));

    int total_threads_temp = rows * cols;
    int num_blocks_temp = (total_threads_temp + threads_per_block - 1) / threads_per_block;
    matmul_kernel_1d<<<num_blocks_temp, threads_per_block>>>(d_H_row, d_C, d_temp, rows, cols, rows);

    cudaMalloc(d_deltaW, rows * cols * sizeof(float));
    int total_threads_final = rows * cols;
    int num_blocks_final = (total_threads_final + threads_per_block - 1) / threads_per_block;
    matmul_kernel_1d<<<num_blocks_final, threads_per_block>>>(d_temp, d_H_col, *d_deltaW, rows, cols, cols);

    cudaFree(d_temp);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <ct_mat_rows> <ct_mat_cols> <num_tokens> <threads_per_block>"
                  << " <ct_directory> <locs_directory> <x_directory>\n";
        return 1;
    }

    int ct_mat_rows_input = std::atoi(argv[1]);
    int ct_mat_cols_input = std::atoi(argv[2]);
    int num_tokens = std::atoi(argv[3]);
    int threads_per_block = std::atoi(argv[4]);
    std::string ct_directory = argv[5];
    std::string locs_directory = argv[6];
    std::string x_directory = argv[7];

    int ct_mat_rows = next_power_of_2(ct_mat_rows_input);
    int ct_mat_cols = next_power_of_2(ct_mat_cols_input);
    std::cout << "\nct_mat_rows: " << ct_mat_rows << ", ct_mat_cols: " << ct_mat_cols << "\n";

    float* h_y = new float[num_tokens * ct_mat_cols];
    float* h_D = new float[ct_mat_rows * ct_mat_cols];
    float* h_dw = new float[ct_mat_rows * ct_mat_cols];

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float* d_deltaW = nullptr;
    calculate_deltaW(d_H_row, d_C, d_H_col, ct_mat_rows, ct_mat_cols, &d_deltaW, threads_per_block);
    std::cout << "\n Delta W completed\n";

    float* d_deltaW_trimmed;
    cudaMalloc(&d_deltaW_trimmed, ct_mat_rows * x_rows * sizeof(float));
    cudaMemcpy2D(d_deltaW_trimmed, x_rows * sizeof(float), d_deltaW, ct_mat_cols * sizeof(float), x_rows * sizeof(float), ct_mat_rows, cudaMemcpyDeviceToDevice);

    int total_threads_Y = x_rows * ct_mat_cols;
    int num_blocks_Y = (total_threads_Y + threads_per_block - 1) / threads_per_block;

    std::cout << "\n Launching 1D matmul_kernel with " << num_blocks_Y << " blocks and " << threads_per_block << " threads per block.\n";

    matmul_kernel_1d<<<num_blocks_Y, threads_per_block>>>(d_deltaW_trimmed, d_x, d_y, ct_mat_rows, x_cols, x_rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\n✅ All computations done on CUDA!\n";
    std::cout << "⏱ CUDA Execution Time: " << milliseconds << " ms\n";

    cudaFree(d_H_row);
    cudaFree(d_H_col);
    cudaFree(d_C);
    cudaFree(d_deltaW);
    cudaFree(d_deltaW_trimmed);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_y;
    delete[] h_D;
    delete[] h_dw;
    delete[] ct_mat;
    delete[] ct;
    delete[] locs;
    delete[] x;

    return 0;
}
