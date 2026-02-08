#include "cg/cg.hpp"

#include "cg/linalg.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cg {

CGResult solve_conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options,
    const std::vector<double>& x0) {
    if (!A.is_valid()) {
        throw std::invalid_argument("solve_conjugate_gradient: invalid matrix");
    }
    if (A.rows != A.cols) {
        throw std::invalid_argument("solve_conjugate_gradient: A must be square");
    }
    if (b.size() != static_cast<std::size_t>(A.rows)) {
        throw std::invalid_argument("solve_conjugate_gradient: b size mismatch");
    }
    if (!x0.empty() && x0.size() != static_cast<std::size_t>(A.cols)) {
        throw std::invalid_argument("solve_conjugate_gradient: x0 size mismatch");
    }

    CGResult result;
    result.x = x0.empty() ? std::vector<double>(A.cols, 0.0) : x0;

    std::vector<double> r = linalg::subtract(b, A.multiply(result.x));
    std::vector<double> p = r;

    double rs_old = linalg::dot(r, r);
    result.residual_norm = std::sqrt(rs_old);
    if (result.residual_norm <= options.tolerance) {
        result.converged = true;
        return result;
    }

    for (std::size_t k = 0; k < options.max_iterations; ++k) {
        const std::vector<double> Ap = A.multiply(p);
        const double denom = linalg::dot(p, Ap);

        if (std::abs(denom) <= std::numeric_limits<double>::epsilon()) {
            result.iterations = k;
            result.residual_norm = std::sqrt(rs_old);
            result.converged = false;
            return result;
        }

        const double alpha = rs_old / denom;
        linalg::axpy(result.x, alpha, p);
        linalg::axpy(r, -alpha, Ap);

        const double rs_new = linalg::dot(r, r);
        result.iterations = k + 1;
        result.residual_norm = std::sqrt(rs_new);
        if (result.residual_norm <= options.tolerance) {
            result.converged = true;
            return result;
        }

        const double beta = rs_new / rs_old;
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    result.converged = false;
    return result;
}

}  // namespace cg
