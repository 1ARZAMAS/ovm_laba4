#include <iostream>
#include <vector>
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void dgemm(int n, const vector<vector<double>>& A,
           const vector<vector<double>>& B,
           vector<vector<double>>& C,
           double alpha = 1.0, double beta = 0.0,
           int threads = 1)
{
#ifdef _OPENMP
    // задаём количество потоков, если указано >0
    if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {            // строки A
        for (int j = 0; j < n; ++j) {        // столбцы B
            double sum = 0.0;
            for (int k = 0; k < n; ++k)      // общий индекс
                sum += A[i][k] * B[k][j];
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    auto begin = chrono::steady_clock::now();

    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <размер> [потоки]" << endl;
        return 1;
    }
    int n        = atoi(argv[1]);
    int threads  = (argc >= 3) ? atoi(argv[2]) : 1;

    if (n <= 0 || threads <= 0) {
        cout << "Размер и число потоков должны быть положительными" << endl;
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr)));

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<double>(rand()) / RAND_MAX;
            B[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }

    dgemm(n, A, B, C, 1.0, 0.0, threads);

    auto end = chrono::steady_clock::now();
    cout << "N=" << n << ", threads=" << threads
         << ", time=" << chrono::duration<double>(end - begin).count()
         << " sec" << endl;
    return 0;
}
