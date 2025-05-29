# Matrix Multiplication Bench

Compare MatMul performance of various library implementations.

## Results

| Test Case |    Library     | Language  |  Result  |
|-----------|----------------|-----------|----------|
|  Input A  |     Eigen      |    C++    |  0.477s  |
|           | matrixmultiply |   Rust    |  0.644s  |
|           |     CBLAS      |   Rust    |  1.226s  |
|  Input B  |     Eigen      |    C++    |  0.497s  |
|           | matrixmultiply |   Rust    |  0.754s  |
|           |     CBLAS      |   Rust    |  1.276s  |
|  Input C  |     Eigen      |    C++    |  2.083s  |
|           | matrixmultiply |   Rust    |  2.783s  |
|           |     CBLAS      |   Rust    |  5.938s  |
|  Input D  |     Eigen      |    C++    | 39.814s  |
|           | matrixmultiply |   Rust    | 45.238s  |
|           |     CBLAS      |   Rust    | 85.979s  |