#include "cg/cg.hpp"
#include "cg/sparse.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchOptions {
    int grid = 256;
    std::size_t repeats = 5;
    std::size_t warmup = 1;
    std::size_t max_iterations = 2000;
    double tolerance = 1e-8;
    std::string mode = "both";
};

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --grid <int>         Grid width for 2D Poisson matrix (default 256)\n"
        << "  --repeats <int>      Timed runs per backend (default 5)\n"
        << "  --warmup <int>       Warmup runs per backend (default 1)\n"
        << "  --max-iters <int>    CG max iterations (default 2000)\n"
        << "  --tol <float>        CG tolerance (default 1e-8)\n"
        << "  --mode <cpu|gpu|both> Backend selection (default both)\n"
        << "  --help               Show this help\n";
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--grid") {
            opts.grid = std::atoi(need_value("--grid"));
        } else if (arg == "--repeats") {
            opts.repeats = static_cast<std::size_t>(std::strtoull(
                need_value("--repeats"), nullptr, 10));
        } else if (arg == "--warmup") {
            opts.warmup = static_cast<std::size_t>(std::strtoull(
                need_value("--warmup"), nullptr, 10));
        } else if (arg == "--max-iters") {
            opts.max_iterations = static_cast<std::size_t>(std::strtoull(
                need_value("--max-iters"), nullptr, 10));
        } else if (arg == "--tol") {
            opts.tolerance = std::atof(need_value("--tol"));
        } else if (arg == "--mode") {
            opts.mode = need_value("--mode");
        } else if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (opts.grid <= 0) {
        throw std::invalid_argument("--grid must be > 0");
    }
    if (opts.repeats == 0) {
        throw std::invalid_argument("--repeats must be > 0");
    }
    if (opts.mode != "cpu" && opts.mode != "gpu" && opts.mode != "both") {
        throw std::invalid_argument("--mode must be one of: cpu, gpu, both");
    }
    return opts;
}

cg::CSRMatrix make_poisson_2d_matrix(int grid) {
    const int n = grid * grid;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    row_ptr.reserve(static_cast<std::size_t>(n) + 1);
    col_idx.reserve(static_cast<std::size_t>(n) * 5);
    values.reserve(static_cast<std::size_t>(n) * 5);

    row_ptr.push_back(0);
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            const int idx = r * grid + c;

            if (r > 0) {
                col_idx.push_back(idx - grid);
                values.push_back(-1.0);
            }
            if (c > 0) {
                col_idx.push_back(idx - 1);
                values.push_back(-1.0);
            }

            col_idx.push_back(idx);
            values.push_back(4.0);

            if (c + 1 < grid) {
                col_idx.push_back(idx + 1);
                values.push_back(-1.0);
            }
            if (r + 1 < grid) {
                col_idx.push_back(idx + grid);
                values.push_back(-1.0);
            }
            row_ptr.push_back(static_cast<int>(values.size()));
        }
    }

    return cg::CSRMatrix(n, n, std::move(row_ptr), std::move(col_idx), std::move(values));
}

struct TimedResult {
    cg::CGResult result;
    double elapsed_ms = 0.0;
};

template <typename SolveFn>
TimedResult run_once(SolveFn&& solve) {
    const auto start = std::chrono::steady_clock::now();
    cg::CGResult result = solve();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return TimedResult{std::move(result), elapsed_ms};
}

struct Stats {
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double median_ms = 0.0;
    cg::CGResult last_result;
};

template <typename SolveFn>
Stats benchmark_backend(
    const std::string& label,
    std::size_t warmup,
    std::size_t repeats,
    SolveFn&& solve) {
    for (std::size_t i = 0; i < warmup; ++i) {
        (void)run_once(solve);
    }

    std::vector<double> times_ms;
    times_ms.reserve(repeats);
    cg::CGResult last_result;
    for (std::size_t i = 0; i < repeats; ++i) {
        const TimedResult run = run_once(solve);
        times_ms.push_back(run.elapsed_ms);
        last_result = run.result;
    }

    const auto [min_it, max_it] = std::minmax_element(times_ms.begin(), times_ms.end());
    const double min_ms = *min_it;
    const double max_ms = *max_it;
    const double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    std::sort(times_ms.begin(), times_ms.end());
    const double median = times_ms[times_ms.size() / 2];

    Stats stats;
    stats.mean_ms = sum / static_cast<double>(times_ms.size());
    stats.min_ms = min_ms;
    stats.max_ms = max_ms;
    stats.median_ms = median;
    stats.last_result = std::move(last_result);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << label << " mean/min/max/median (ms): " << stats.mean_ms << " / "
              << stats.min_ms << " / " << stats.max_ms << " / " << stats.median_ms
              << '\n';
    std::cout << label << " converged=" << std::boolalpha << stats.last_result.converged
              << ", iterations=" << stats.last_result.iterations
              << ", residual=" << std::setprecision(8) << stats.last_result.residual_norm
              << '\n';

    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchOptions opts = parse_args(argc, argv);
        const cg::CSRMatrix A = make_poisson_2d_matrix(opts.grid);
        const std::vector<double> b(static_cast<std::size_t>(A.rows), 1.0);
        const cg::CGOptions cg_opts{opts.max_iterations, opts.tolerance};

        std::cout << "Matrix: 2D Poisson " << opts.grid << "x" << opts.grid
                  << " (n=" << A.rows << ", nnz=" << A.values.size() << ")\n";
        std::cout << "Runs: warmup=" << opts.warmup << ", repeats=" << opts.repeats
                  << ", max_iterations=" << cg_opts.max_iterations
                  << ", tolerance=" << cg_opts.tolerance << "\n\n";

        bool ran_cpu = false;
        bool ran_gpu = false;
        Stats cpu_stats{};
        Stats gpu_stats{};

        if (opts.mode == "cpu" || opts.mode == "both") {
            cpu_stats = benchmark_backend(
                "CPU",
                opts.warmup,
                opts.repeats,
                [&]() { return cg::solve_conjugate_gradient(A, b, cg_opts); });
            ran_cpu = true;
            std::cout << '\n';
        }

        if (opts.mode == "gpu" || opts.mode == "both") {
            if (!cg::gpu_backend_available()) {
                std::cout << "GPU unavailable at runtime.\n";
            } else {
                gpu_stats = benchmark_backend(
                    "GPU",
                    opts.warmup,
                    opts.repeats,
                    [&]() { return cg::solve_conjugate_gradient_gpu(A, b, cg_opts); });
                ran_gpu = true;
            }
        }

        if (ran_cpu && ran_gpu) {
            std::cout << '\n';
            const double speedup = cpu_stats.mean_ms / gpu_stats.mean_ms;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Speedup (CPU mean / GPU mean): " << speedup << "x\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
