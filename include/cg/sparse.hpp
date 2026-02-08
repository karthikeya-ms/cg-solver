#pragma once

#include <vector>

namespace cg {

struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;

    CSRMatrix() = default;
    CSRMatrix(
        int rows_,
        int cols_,
        std::vector<int> row_ptr_,
        std::vector<int> col_idx_,
        std::vector<double> values_);

    bool is_valid() const;
    std::vector<double> multiply(const std::vector<double>& x) const;
};

}  // namespace cg
