#include <benchmark/benchmark.h>
#include <cstdlib>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#ifndef THREADS
#define THREADS 1
#endif

struct MatmulInputs {
    size_t out_size;
    size_t inp_size;
    size_t weight_size;
    size_t B;
    size_t T;
    size_t C;
    size_t OC;
};

// Tensor types like in TensorFlow
template <typename T>
using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>, Eigen::Aligned>;
template <typename T>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>;

extern "C" {
    void matmul_matrixmultiply(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
    void matmul_cblas(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
}

void matmul_eigen(Matrix<double> y, Eigen::ThreadPoolDevice device, ConstMatrix<double> x1, ConstMatrix<double> x2, Eigen::array<Eigen::IndexPair<int>, 1> dims) {
    y.device(device) = x1.contract(x2, dims);
}

std::vector<MatmulInputs> configs = {
    {196608, 786432, 2359296, 4, 64, 3072, 768},
    {196608, 196608, 589824, 4, 64, 768, 768},
    {786432, 196608, 2359296, 4, 64, 768, 3072},
    {12877824, 196608, 38633472, 4, 64, 768, 50304},
};

static void BM_Eigen(benchmark::State& state) {
    const MatmulInputs& cfg = configs[state.range(0)];

    int rep = std::max<int>(1, 10000000 / cfg.B / cfg.T / cfg.C);

    std::vector<double> a(cfg.inp_size);
    std::vector<double> b(cfg.weight_size);
    std::vector<double> c(cfg.out_size);

    for (auto& v : a) v = static_cast<double>(rand()) / RAND_MAX;
    for (auto& v : b) v = static_cast<double>(rand()) / RAND_MAX;

    ConstMatrix<double> x1(a.data(), cfg.B * cfg.T, cfg.C);
    ConstMatrix<double> x2(b.data(), cfg.C, cfg.OC);
    Matrix<double> y(c.data(), cfg.B * cfg.T, cfg.OC);

    Eigen::ThreadPool g_thread_pool(THREADS);
    Eigen::ThreadPoolDevice g_device(&g_thread_pool, THREADS);

    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};

    for (auto _ : state) {
        for (int i = 0; i < rep; ++i) {
            matmul_eigen(y, g_device, x1, x2, dims);
        }
    }
}
BENCHMARK(BM_Eigen)->Args({0});
BENCHMARK(BM_Eigen)->Args({1});
BENCHMARK(BM_Eigen)->Args({2});
BENCHMARK(BM_Eigen)->Args({3});

static void BM_MatrixMultiply(benchmark::State& state) {
    const MatmulInputs& cfg = configs[state.range(0)];

    int rep = std::max<int>(1, 10000000 / cfg.B / cfg.T / cfg.C);

    std::vector<double> a(cfg.inp_size);
    std::vector<double> b(cfg.weight_size);
    std::vector<double> c(cfg.out_size);

    for (auto& v : a) v = static_cast<double>(rand()) / RAND_MAX;
    for (auto& v : b) v = static_cast<double>(rand()) / RAND_MAX;

    for (auto _ : state) {
        for (int i = 0; i < rep; ++i) {
            matmul_matrixmultiply(c.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC);
        }
    }
}
BENCHMARK(BM_MatrixMultiply)->Args({0});
BENCHMARK(BM_MatrixMultiply)->Args({1});
BENCHMARK(BM_MatrixMultiply)->Args({2});
BENCHMARK(BM_MatrixMultiply)->Args({3});

static void BM_CBLAS(benchmark::State& state) {
    const MatmulInputs& cfg = configs[state.range(0)];

    int rep = std::max<int>(1, 10000000 / cfg.B / cfg.T / cfg.C);

    std::vector<double> a(cfg.inp_size);
    std::vector<double> b(cfg.weight_size);
    std::vector<double> c(cfg.out_size);

    for (auto& v : a) v = static_cast<double>(rand()) / RAND_MAX;
    for (auto& v : b) v = static_cast<double>(rand()) / RAND_MAX;

    for (auto _ : state) {
        for (int i = 0; i < rep; ++i) {
            matmul_cblas(c.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC);
        }
    }
}
BENCHMARK(BM_CBLAS)->Args({0});
BENCHMARK(BM_CBLAS)->Args({1});
BENCHMARK(BM_CBLAS)->Args({2});
BENCHMARK(BM_CBLAS)->Args({3});

int main(int argc, char* argv[]) {
    setenv("MKL_NUM_THREADS", std::to_string(THREADS).c_str(), 1);
    setenv("OMP_NUM_THREADS", std::to_string(THREADS).c_str(), 1);
    setenv("MATMUL_NUM_THREADS", std::to_string(THREADS).c_str(), 1);

    if (THREADS > 4) {
        std::cout << "WARNING: " << THREADS << " threads specified, but matrixmultiply supports a maximum of 4 threads. Using 4 threads instead!" << std::endl;
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}