#include<iostream>
#include<unsupported/Eigen/CXX11/Tensor>
#include<bench/BenchTimer.h>

// Tensor types like in TensorFlow
template <typename T>
using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>, Eigen::Aligned>;
template <typename T>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>;

extern "C" {
    void matmul(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
}

bool compare_results(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tol = 1e-8) {
    return ((m1 - m2).array().abs() < tol).all();
}

int main(int argc, char* argv[]) {
    constexpr int N = 5000;
    constexpr int tries = 4;
    int rep = std::max<int>(1, 10000000 / N / N / N);

    std::vector<double> a(N * N);
    std::vector<double> b(N * N);
    std::vector<double> c(N * N);

    // Fill input matrices with random values
    for (auto& v : a) v = static_cast<double>(rand()) / RAND_MAX;
    for (auto& v : b) v = static_cast<double>(rand()) / RAND_MAX;

    ConstMatrix<double> x1(a.data(), N, N);
    ConstMatrix<double> x2(b.data(), N, N);
    Matrix<double> y(c.data(), N, N);

    Eigen::DefaultDevice device;

    Eigen::BenchTimer t1, t2;

    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};

    BENCH(t1, tries, rep, y.device(device) = x1.contract(x2, dims));

    BENCH(t2, tries, rep, matmul(c.data(), a.data(), b.data(), 1, N, N, N));

    std::cout << "Time taken by Eigen Tensor contraction: " << t1.best() << "\n";
    std::cout << "Time taken by Rust matmul: " << t2.best() << "\n";

    return 0;
}