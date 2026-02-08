#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cg::linalg {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vectors must have the same size");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline double norm2(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

inline void axpy(
    std::vector<double>& y,
    double alpha,
    const std::vector<double>& x) {
    if (y.size() != x.size()) {
        throw std::invalid_argument("axpy: vectors must have the same size");
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

inline std::vector<double> subtract(
    const std::vector<double>& a,
    const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("subtract: vectors must have the same size");
    }

    std::vector<double> out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

}  // namespace cg::linalg
