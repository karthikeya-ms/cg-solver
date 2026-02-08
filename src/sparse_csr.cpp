#include "cg/sparse.hpp"

#include <stdexcept>

namespace cg {

CSRMatrix::CSRMatrix(
    int rows_,
    int cols_,
    std::vector<int> row_ptr_,
    std::vector<int> col_idx_,
    std::vector<double> values_)
    : rows(rows_),
      cols(cols_),
      row_ptr(std::move(row_ptr_)),
      col_idx(std::move(col_idx_)),
      values(std::move(values_)) {}

bool CSRMatrix::is_valid() const {
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    if (row_ptr.size() != static_cast<std::size_t>(rows + 1)) {
        return false;
    }
    if (row_ptr.front() != 0) {
        return false;
    }
    if (col_idx.size() != values.size()) {
        return false;
    }
    if (row_ptr.back() != static_cast<int>(values.size())) {
        return false;
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i) {
        if (row_ptr[i] < row_ptr[i - 1]) {
            return false;
        }
    }
    for (int c : col_idx) {
        if (c < 0 || c >= cols) {
            return false;
        }
    }
    return true;
}

std::vector<double> CSRMatrix::multiply(const std::vector<double>& x) const {
    if (!is_valid()) {
        throw std::invalid_argument("CSRMatrix::multiply: invalid matrix");
    }
    if (x.size() != static_cast<std::size_t>(cols)) {
        throw std::invalid_argument("CSRMatrix::multiply: dimension mismatch");
    }

    std::vector<double> y(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            y[r] += values[k] * x[col_idx[k]];
        }
    }
    return y;
}

}  // namespace cg
