#include "cg/cg.hpp"
#include "cg/sparse.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const cg::CSRMatrix A(
        3,
        3,
        {0, 2, 5, 7},
        {0, 1, 0, 1, 2, 1, 2},
        {4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0});

    const std::vector<double> b = {6.0, 10.0, 8.0};
    const cg::CGOptions options{100, 1e-12};

    const cg::CGResult result = cg::solve_conjugate_gradient(A, b, options);

    std::cout << std::boolalpha;
    std::cout << "Converged: " << result.converged << '\n';
    std::cout << "Iterations: " << result.iterations << '\n';
    std::cout << "Residual norm: " << std::setprecision(12) << result.residual_norm
              << '\n';
    std::cout << "x = [";
    for (std::size_t i = 0; i < result.x.size(); ++i) {
        std::cout << result.x[i];
        if (i + 1 != result.x.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    return result.converged ? 0 : 1;
}
