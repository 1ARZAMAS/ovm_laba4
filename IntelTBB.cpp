#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <oneapi/tbb.h>
#include <oneapi/tbb/global_control.h>

using namespace std;
using namespace oneapi::tbb;

void dgemm(int n, const vector<vector<double>>& A, const vector<vector<double>>& B,
           vector<vector<double>>& C, double alpha = 1.0, double beta = 0.0) {

    parallel_for(blocked_range<int>(0, n), [&](const blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = alpha * sum + beta * C[i][j];
            }
        }
    });
}

int main(int argc, char* argv[]) {
    auto begin = chrono::steady_clock::now();

    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <размер> [потоки]" << endl;
        return 1;
    }
    int n = atoi(argv[1]);
    int threads = (argc >= 3) ? atoi(argv[2]) : tbb::info::default_concurrency();
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);

    int n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Ошибка: размер матрицы должен быть положительным числом" << endl;
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

    dgemm(n, A, B, C);

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - begin;
    cout << "Time: " << elapsed.count() << " sec" << endl;

    return 0;
}
