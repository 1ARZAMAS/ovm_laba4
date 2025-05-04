#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define IDX2C(i,j,n) (((i)*(n))+(j))  // макрос для индексации

__global__ void dgemm_kernel(int n, const double* A, const double* B, double* C, double alpha, double beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // строка
    int col = blockIdx.x * blockDim.x + threadIdx.x; // столбец

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[IDX2C(row, k, n)] * B[IDX2C(k, col, n)];
        }
        C[IDX2C(row, col, n)] = alpha * sum + beta * C[IDX2C(row, col, n)];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <размер матрицы> <потоки на блок>"<< std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int threadsPerBlock = (argc >= 3) ? atoi(argv[2]) : 16;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    if (n <= 0) {
        std::cerr << "Ошибка: размер должен быть положительным числом" << std::endl;
        return 1;
    }

    size_t bytes = n * n * sizeof(double);
    std::vector<double> h_A(n * n), h_B(n * n), h_C(n * n, 0.0);

    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    auto begin = std::chrono::steady_clock::now();
    dgemm_kernel<<<blocks, threads>>>(n, d_A, d_B, d_C, 1.0, 0.0);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    std::cout << "Time: " << elapsed.count() << " sec" << std::endl;
    return 0;
}
