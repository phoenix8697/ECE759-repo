
#include <cnpy.h>
#include "load_ckpt.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>

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

// CUDA Matrix Multiplication Kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Calculate deltaW = H * C * H^T
void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col, int rows, int cols, float** d_deltaW) {
    float* d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_H_row, d_C, d_temp, rows, cols, rows);
    cudaDeviceSynchronize();

    cudaMalloc(d_deltaW, rows * cols * sizeof(float));
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_H_col, *d_deltaW, rows, cols, cols);
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

    // CUDA event setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Parse input
    int ct_mat_rows_input = std::atoi(argv[1]);
    int ct_mat_cols_input = std::atoi(argv[2]);
    int num_tokens = std::atoi(argv[3]);
    int threads_per_block = std::atoi(argv[4]);
    std::string ct_directory = argv[5];
    std::string locs_directory = argv[6];
    std::string x_directory = argv[7];

    int ct_mat_rows = next_power_of_2(ct_mat_rows_input);
    int ct_mat_cols = next_power_of_2(ct_mat_cols_input);

    // Allocate host memory
    float* h_y = new float[num_tokens * ct_mat_cols];
    float* h_D = new float[ct_mat_rows * ct_mat_cols];
    float* h_dw = new float[ct_mat_rows * ct_mat_cols];

    // Load coefficient and location data
    float* ct = nullptr;
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    int* locs = nullptr;
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

    float* x = nullptr;
    auto [x_rows, x_cols] = load_ckpt_float(x_directory, x);

    // Construct full dense matrix from sparse representation
    float* ct_mat = new float[ct_mat_rows * ct_mat_cols]();
    for (size_t i = 0; i < locs_cols; ++i) {
        int row = locs[i];
        int col = locs[locs_cols + i];
        ct_mat[row * ct_mat_cols + col] = ct[i];
    }

    // Start timing
    cudaEventRecord(start);

    // Allocate and copy C matrix to device
    float* d_C = nullptr;
    cudaMalloc(&d_C, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMemcpy(d_C, ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create Hadamard matrices
    float* d_H_row = nullptr;
    float* d_H_col = nullptr;
    create_hadamard_matrix(ct_mat_rows, &d_H_row);
    create_hadamard_matrix(ct_mat_cols, &d_H_col);

    // Copy input X to device
    float* d_x = nullptr;
    cudaMalloc(&d_x, x_rows * x_cols * sizeof(float));
    cudaMemcpy(d_x, x, x_rows * x_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate output Y on device
    float* d_y = nullptr;
    cudaMalloc(&d_y, x_rows * ct_mat_cols * sizeof(float));

    

    // deltaW = H × C × Hᵗ
    float* d_deltaW = nullptr;
    calculate_deltaW(d_H_row, d_C, d_H_col, ct_mat_rows, ct_mat_cols, &d_deltaW);
    cudaDeviceSynchronize();

    // Y = deltaW × X
    dim3 threadsPerBlockDim(16, 16);
    dim3 blocksPerGrid((ct_mat_cols + 15) / 16, (x_rows + 15) / 16);
    // last parameter to matmul_kernel is truncated to x_cols truncate the ct_mat_cols to x_cols
    //int x_cols_compute = ct_mat_rows > x_cols ? x_cols : ct_mat_rows;
    matmul_kernel<<<blocksPerGrid, threadsPerBlockDim>>>(d_deltaW, d_x, d_y, x_rows, ct_mat_cols, x_cols); 
    cudaDeviceSynchronize();

    // End timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "\n All computations done on CUDA!\n";
    std::cout << "CUDA Execution Time: " << milliseconds << " ms\n";

    // Cleanup
    cudaFree(d_H_row);
    cudaFree(d_H_col);
    cudaFree(d_C);
    cudaFree(d_deltaW);
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
