# cg-solver

Conjugate Gradient solver in C++.

## Build

```bash
cmake -S . -B build
cmake --build build
```

If a CUDA toolkit/compiler is available, CMake automatically enables a GPU
implementation of the solver.
The GPU implementation uses cuSPARSE (CSR SpMV) and cuBLAS (vector ops).
If cuBLAS/cuSPARSE development headers or libraries are missing, build falls
back to a GPU stub.
Default CUDA architecture is set to `75`; override with
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` if needed.

## Run

```bash
./build/cg_driver
```
