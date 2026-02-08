#include "cg/cg.hpp"

#include <stdexcept>

namespace cg {

bool gpu_backend_available() {
    return false;
}

CGResult solve_conjugate_gradient_gpu(
    const CSRMatrix&,
    const std::vector<double>&,
    const CGOptions&,
    const std::vector<double>&) {
    throw std::runtime_error(
        "solve_conjugate_gradient_gpu: CUDA support is not enabled in this build");
}

}  // namespace cg
