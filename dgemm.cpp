#include <iostream>
#include <vector>
#include <cstdlib> // для rand() и srand()
#include <ctime> // для time()
#include <chrono>

using namespace std;

void dgemm(int n, const vector<vector<double>>& A, const vector<vector<double>>& B, 
    vector<vector<double>>& C, double alpha = 1.0, double beta = 0.0) {
    for (int i = 0; i < n; ++i) {         // строки A
        for (int j = 0; j < n; ++j) {     // столбцы B
            double sum = 0.0;
            for (int k = 0; k < n; ++k) { // общий индекс
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    auto begin = std::chrono::steady_clock::now();
    if (argc != 2) {
        cout << "Использование: " << argv[0] << " размер матрицы" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        cout << "Ошибка: размер матрицы должен быть положительным числом" << endl;
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr))); // инициализация генератора случайных чисел

    // Создание матриц
    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    // Заполнение матриц случайными числами от 0 до 1
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<double>(rand()) / RAND_MAX;
            B[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }

    // Вызов функции DGEMM
    dgemm(n, A, B, C);
    auto end = chrono::steady_clock::now();
    auto elapsed_ns = chrono::duration_cast<chrono::duration<double>>(end - begin);
    cout << "Time: " << elapsed_ns.count() << " sec" << endl;
    return 0;
}
