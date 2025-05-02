#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>

using namespace std;

void dgemm_local(int start_row, int end_row, int n,
                 const vector<vector<double>>& A, const vector<vector<double>>& B,
                 vector<vector<double>>& C, double alpha = 1.0, double beta = 0.0) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            cout << "Использование: " << argv[0] << " размер_матрицы" << endl;
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0)
            cout << "Ошибка: размер матрицы должен быть положительным числом" << endl;
        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> A, B, C;

    auto start_time = MPI_Wtime();

    if (rank == 0) {
        srand(static_cast<unsigned>(time(nullptr)));
        A.resize(n, vector<double>(n));
        B.resize(n, vector<double>(n));
        C.resize(n, vector<double>(n, 0.0));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                A[i][j] = static_cast<double>(rand()) / RAND_MAX;
                B[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
    }

    // Рассылаем A и B всем процессам (broadcast)
    if (rank != 0) {
        A.resize(n, vector<double>(n));
        B.resize(n, vector<double>(n));
        C.resize(n, vector<double>(n, 0.0));
    }

    for (int i = 0; i < n; ++i) {
        MPI_Bcast(&A[i][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[i][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

    dgemm_local(start_row, end_row, n, A, B, C);

    // Сборка результата в процесс 0
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            int s = r * rows_per_proc + min(r, remainder);
            int e = s + rows_per_proc + (r < remainder ? 1 : 0);
            for (int i = s; i < e; ++i) {
                MPI_Recv(&C[i][0], n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(&C[i][0], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        auto end_time = MPI_Wtime();
        cout << "Time: " << (end_time - start_time) << " sec" << endl;
    }

    MPI_Finalize();
    return 0;
}
