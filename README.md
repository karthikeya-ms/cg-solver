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

## Benchmark

```bash
./build/cg_benchmark --grid 256 --repeats 5 --warmup 1 --mode both
```

Key options:
- `--mode cpu|gpu|both`
- `--grid <N>` where matrix size is `(N*N) x (N*N)`
- `--repeats <R>` and `--warmup <W>`
- `--max-iters <K>` and `--tol <eps>`

## Profiling With Nsight Systems

Create a profile output directory:

```bash
mkdir -p profiles
```

CPU profile (sampling-focused):

```bash
nsys profile \
  --output=profiles/cg_cpu \
  --force-overwrite=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --trace=osrt \
  ./build/cg_benchmark --mode cpu --grid 256 --warmup 1 --repeats 5
```

GPU profile (CUDA + cuBLAS + cuSPARSE timeline):

```bash
nsys profile \
  --output=profiles/cg_gpu \
  --force-overwrite=true \
  --trace=cuda,cublas,cusparse,osrt \
  ./build/cg_benchmark --mode gpu --grid 256 --warmup 1 --repeats 5
```

Quick CLI summary from generated reports:

```bash
nsys stats --report osrt_sum profiles/cg_cpu.nsys-rep
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum profiles/cg_gpu.nsys-rep
```

## Inspect In Wafer (VS Code)

1. Open VS Code in this workspace.
2. Open the Command Palette and run `Wafer: Open Trace`.
3. Select `profiles/cg_cpu.nsys-rep` or `profiles/cg_gpu.nsys-rep`.
4. In the timeline:
   - CPU run: inspect sampled stacks and long CPU regions.
   - GPU run: inspect CUDA API calls, cuBLAS/cuSPARSE calls, and kernel overlap.
