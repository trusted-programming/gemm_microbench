#include<iostream>
#include<unsupported/Eigen/CXX11/Tensor>
#include<bench/BenchTimer.h>

// Tensor types like in TensorFlow
template <typename T>
using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>, Eigen::Aligned>;
template <typename T>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>;

struct MatmulInputs {
    size_t out_size;
    size_t inp_size;
    size_t weight_size;
    size_t B;
    size_t T;
    size_t C;
    size_t OC;
};

extern "C" {
    void matmul_matrixmultiply(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
    void matmul_cblas(double* out, const double* inp, const double* weight, size_t B, size_t T, size_t C, size_t OC);
}

void matmul_eigen(Matrix<double> y, Eigen::DefaultDevice device, ConstMatrix<double> x1, ConstMatrix<double> x2, Eigen::array<Eigen::IndexPair<int>, 1> dims) {
    y.device(device) = x1.contract(x2, dims);
}

bool compare_results(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tol = 1e-8) {
    return ((m1 - m2).array().abs() < tol).all();
}

int main(int argc, char* argv[]) {
    constexpr int tries = 4;
    
    std::vector<MatmulInputs> configs = {
        {196608, 786432, 2359296, 4, 64, 3072, 768},
        {196608, 196608, 589824, 4, 64, 768, 768},
        {786432, 196608, 2359296, 4, 64, 768, 3072},
        {12877824, 196608, 38633472, 4, 64, 768, 50304},
    };

    for (const auto& cfg : configs) {
        std::cout << "\nRunning configuration: "
                  << "B=" << cfg.B << ", T=" << cfg.T << ", C=" << cfg.C << ", OC=" << cfg.OC << "\n";

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

        Eigen::DefaultDevice device;

        Eigen::BenchTimer t_eigen, t_matrixmultiply, t_cblas;

        Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};

        BENCH(t_eigen, tries, rep, matmul_eigen(y, device, x1, x2, dims));
        BENCH(t_matrixmultiply, tries, rep, matmul_matrixmultiply(c_matrixmultiply.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC));
        BENCH(t_cblas, tries, rep, matmul_cblas(c_cblas.data(), a.data(), b.data(), cfg.B, cfg.T, cfg.C, cfg.OC));

        std::cout << "Time taken by Eigen: " << t_eigen.best() << "\n";
        std::cout << "Time taken by matrixmultiply: " << t_matrixmultiply.best() << "\n";
        std::cout << "Time taken by CBLAS: " << t_cblas.best() << "\n";

        // Comparison
        double tol = 1e-8;
        bool match = true;
        size_t total_elements = cfg.B * cfg.T * cfg.OC;

        for (size_t i = 0; i < total_elements; ++i) {
            if (std::abs(c_eigen[i] - c_matrixmultiply[i]) > tol || 
                std::abs(c_eigen[i] - c_cblas[i]) > tol) {
                std::cout << "Mismatch at index " << i
                        << ": Eigen=" << c_eigen[i]
                        << ", matrixmultiply=" << c_matrixmultiply[i] 
                        << ", CBLAS=" << c_cblas[i] << "\n";
                match = false;
                break;
            }
        }

        if (match) {
            std::cout << "The results MATCH!\n";
        } else {
            std::cout << "The results are DIFFERENT!\n";
        }
    }

    return 0;
}