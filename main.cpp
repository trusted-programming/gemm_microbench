#include<iostream>
#include<cstddef>
#include<Eigen/Dense>
#include<bench/BenchTimer.h>

extern "C" {
    void matmul(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
}

int main(int argc, char* argv[]) {
    int N = 5000;
    int tries = 4;
    int rep = std::max<int>(1,10000000/N/N/N);

    Eigen::MatrixXd a_E = Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd b_E = Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd c_E(N,N);

    Eigen::BenchTimer t1, t2;

    BENCH(t1, tries, rep, c_E.noalias() = a_E*b_E );
    BENCH(t2, tries, rep, matmul(c_E.data(), a_E.data(), b_E.data(), 1, 1, N, N));
    
    std::cout << "Time taken by Eigen (C++): " << t1.best() << "\n";
    std::cout << "Time taken by matrixmultiply (Rust): " << t2.best() << "\n\n";
}