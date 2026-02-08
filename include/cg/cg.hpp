#pragma once

#include "cg/sparse.hpp"

#include <cstddef>
#include <vector>

namespace cg {

struct CGOptions {
    std::size_t max_iterations = 1000;
    double tolerance = 1e-10;
};

struct CGResult {
    std::vector<double> x;
    std::size_t iterations = 0;
    double residual_norm = 0.0;
    bool converged = false;
};

CGResult solve_conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options = {},
    const std::vector<double>& x0 = {});

}  // namespace cg
