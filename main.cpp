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

constexpr double tol = 1e-8;

void check_correctness(const std::vector<double>& c_eigen,
                       const std::vector<double>& c_matrixmultiply,
                       const std::vector<double>& c_cblas,
                       size_t total_elements) {
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(c_eigen[i] - c_matrixmultiply[i]) > tol ||
            std::abs(c_eigen[i] - c_cblas[i]) > tol) {
            std::cerr << "MISMATCH at index " << i
                      << ": Eigen=" << c_eigen[i]
                      << ", matrixmultiply=" << c_matrixmultiply[i]
                      << ", CBLAS=" << c_cblas[i] << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

static void prepare_and_check(const MatmulInputs& cfg) {
    int rep = std::max<int>(1, 10000000 / cfg.B / cfg.T / cfg.C);

    std::vector<double> a(cfg.inp_size);
    std::vector<double> b(cfg.weight_size);
    std::vector<double> c_eigen(cfg.out_size);
    std::vector<double> c_matrixmultiply(cfg.out_size);
    std::vector<double> c_cblas(cfg.out_size);

    for (auto& v : a) v = static_cast<double>(rand()) / RAND_MAX;
    for (auto& v : b) v = static_cast<double>(rand()) / RAND_MAX;

    ConstMatrix<double> x1(a.data(), cfg.B * cfg.T, cfg.C);
    ConstMatrix<double> x2(b.data(), cfg.C, cfg.OC);
    Matrix<double> y(c_eigen.data(), cfg.B * cfg.T, cfg.OC);

    Eigen::ThreadPool g_thread_pool(THREADS);
    Eigen::ThreadPoolDevice g_device(&g_thread_pool, THREADS);

    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};

    // Run once each to fill output buffers
    matmul_eigen(y, g_device, x1, x2, dims);
    matmul_matrixmultiply(c_matrixmultiply.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC);
    matmul_cblas(c_cblas.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC);

    check_correctness(c_eigen, c_matrixmultiply, c_cblas, cfg.B * cfg.T * cfg.OC);
}

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

int main(int argc, char* argv[]) {
    setenv("MKL_NUM_THREADS", std::to_string(THREADS).c_str(), 1);
    setenv("OMP_NUM_THREADS", std::to_string(THREADS).c_str(), 1);
    setenv("MATMUL_NUM_THREADS", std::to_string(THREADS).c_str(), 1);

    if (THREADS > 4) {
        std::cout << "***WARNING*** " << THREADS << " threads specified, but matrixmultiply supports a maximum of 4 threads. Using 4 threads instead!\n" << std::endl;
    }

    // Check correctness for each config once before benchmarking
    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "Checking correctness for config " << i << "..." << std::endl;
        prepare_and_check(configs[i]);
        std::cout << "Results MATCH!\n" << std::endl;
    }

    BENCHMARK(BM_Eigen)->Args({0})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_Eigen)->Args({1})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_Eigen)->Args({2})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_Eigen)->Args({3})->Unit(benchmark::kMillisecond);

    BENCHMARK(BM_MatrixMultiply)->Args({0})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_MatrixMultiply)->Args({1})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_MatrixMultiply)->Args({2})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_MatrixMultiply)->Args({3})->Unit(benchmark::kMillisecond);

    BENCHMARK(BM_CBLAS)->Args({0})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_CBLAS)->Args({1})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_CBLAS)->Args({2})->Unit(benchmark::kMillisecond);
    BENCHMARK(BM_CBLAS)->Args({3})->Unit(benchmark::kMillisecond);

    benchmark::RunSpecifiedBenchmarks();

    return 0;
}