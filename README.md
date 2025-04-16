# Matrix Multiplication Bench

Compare MatMul performance of various library implementations.

## Results

|    Library     | Language | Size  | Result  |
|----------------|----------|-------|---------|
|     Eigen      |   C++    |  500  | 0.008s  |
|                |          | 1000  | 0.061s  |
|                |          | 5000  | 7.610s  |
|                |          | 10000 | 61.805s |
| matrixmultiply |   Rust   |  500  | 0.010s  |
|                |          | 1000  | 0.076s  |
|                |          | 5000  | 10.640s |
|                |          | 10000 | 86.395s |