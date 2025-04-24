#include <iostream>
#include <vector>
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <chrono>
#include <pthread.h>

using namespace std;

struct WorkerArgs {
    const vector<vector<double>>* A;
    const vector<vector<double>>* B;
    vector<vector<double>>* C;
    int n, row_begin, row_end;
    double alpha, beta;
};

// Поток вычисляет строки [row_begin, row_end)
void* worker(void* arg) {
    auto* w = static_cast<WorkerArgs*>(arg);
    int n = w->n;
    for (int i = w->row_begin; i < w->row_end; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += (*(w->A))[i][k] * (*(w->B))[k][j];
            (*(w->C))[i][j] = w->alpha * sum + w->beta * (*(w->C))[i][j];
        }
    return nullptr;
}

int main(int argc, char* argv[]) {
    auto begin = chrono::steady_clock::now();

    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <размер> [потоки]" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int nthreads = (argc >= 3) ? atoi(argv[2]) : 1;

    if (n <= 0 || nthreads <= 0) {
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

    // --- запуск POSIX‑потоков ---
    vector<pthread_t> tids(nthreads);
    vector<WorkerArgs> args(nthreads);

    int rows_per = n / nthreads;
    int extra = n % nthreads;
    int row = 0;

    for (int t = 0; t < nthreads; ++t) {
        int start = row;
        int count = rows_per + (t < extra ? 1 : 0);
        int end = start + count;

        args[t] = {&A, &B, &C, n, start, end, 1.0, 0.0};
        pthread_create(&tids[t], nullptr, worker, &args[t]);
        row = end;
    }
    for (auto& th : tids) pthread_join(th, nullptr);

    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end - begin).count();
    cout << "N=" << n << ", threads=" << nthreads
         << ", time=" << elapsed << " sec" << endl;
    return 0;
}
