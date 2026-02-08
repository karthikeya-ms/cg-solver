#include "cg/cg.hpp"

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg {
namespace {

inline void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + ": cuBLAS status " + std::to_string(status));
    }
}

inline void check_cusparse(cusparseStatus_t status, const char* what) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + ": cuSPARSE status " + std::to_string(status));
    }
}

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }
    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

class CusparseHandle {
public:
    CusparseHandle() {
        check_cusparse(cusparseCreate(&handle_), "cusparseCreate");
    }
    ~CusparseHandle() {
        if (handle_ != nullptr) {
            cusparseDestroy(handle_);
        }
    }
    CusparseHandle(const CusparseHandle&) = delete;
    CusparseHandle& operator=(const CusparseHandle&) = delete;

    cusparseHandle_t get() const { return handle_; }

private:
    cusparseHandle_t handle_ = nullptr;
};

class CsrDescriptor {
public:
    CsrDescriptor(
        int rows,
        int cols,
        int nnz,
        int* row_ptr,
        int* col_idx,
        double* values) {
        check_cusparse(
            cusparseCreateCsr(
                &descriptor_,
                rows,
                cols,
                nnz,
                row_ptr,
                col_idx,
                values,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F),
            "cusparseCreateCsr");
    }
    ~CsrDescriptor() {
        if (descriptor_ != nullptr) {
            cusparseDestroySpMat(descriptor_);
        }
    }
    CsrDescriptor(const CsrDescriptor&) = delete;
    CsrDescriptor& operator=(const CsrDescriptor&) = delete;

    cusparseSpMatDescr_t get() const { return descriptor_; }

private:
    cusparseSpMatDescr_t descriptor_ = nullptr;
};

class DnVectorDescriptor {
public:
    DnVectorDescriptor(int n, double* data) {
        check_cusparse(
            cusparseCreateDnVec(&descriptor_, n, data, CUDA_R_64F),
            "cusparseCreateDnVec");
    }
    ~DnVectorDescriptor() {
        if (descriptor_ != nullptr) {
            cusparseDestroyDnVec(descriptor_);
        }
    }
    DnVectorDescriptor(const DnVectorDescriptor&) = delete;
    DnVectorDescriptor& operator=(const DnVectorDescriptor&) = delete;

    cusparseDnVecDescr_t get() const { return descriptor_; }

private:
    cusparseDnVecDescr_t descriptor_ = nullptr;
};

inline std::size_t spmv_workspace_size(
    cusparseHandle_t cusparse,
    cusparseSpMatDescr_t A,
    int n,
    double* x,
    double* y) {
    DnVectorDescriptor vec_x(n, x);
    DnVectorDescriptor vec_y(n, y);

    const double alpha = 1.0;
    const double beta = 0.0;
    std::size_t workspace_size = 0;
    check_cusparse(
        cusparseSpMV_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A,
            vec_x.get(),
            &beta,
            vec_y.get(),
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &workspace_size),
        "cusparseSpMV_bufferSize");
    return workspace_size;
}

inline void spmv(
    cusparseHandle_t cusparse,
    cusparseSpMatDescr_t A,
    int n,
    double* x,
    double* y,
    void* workspace) {
    DnVectorDescriptor vec_x(n, x);
    DnVectorDescriptor vec_y(n, y);

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cusparse(
        cusparseSpMV(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A,
            vec_x.get(),
            &beta,
            vec_y.get(),
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace),
        "cusparseSpMV");
}

inline double dot(cublasHandle_t cublas, int n, const double* x, const double* y) {
    double result = 0.0;
    check_cublas(cublasDdot(cublas, n, x, 1, y, 1, &result), "cublasDdot");
    return result;
}

inline void axpy(
    cublasHandle_t cublas,
    int n,
    double alpha,
    const double* x,
    double* y) {
    check_cublas(cublasDaxpy(cublas, n, &alpha, x, 1, y, 1), "cublasDaxpy");
}

inline void copy(cublasHandle_t cublas, int n, const double* x, double* y) {
    check_cublas(cublasDcopy(cublas, n, x, 1, y, 1), "cublasDcopy");
}

inline void scale(cublasHandle_t cublas, int n, double alpha, double* x) {
    check_cublas(cublasDscal(cublas, n, &alpha, x, 1), "cublasDscal");
}

inline double* raw_ptr(thrust::device_vector<double>& v) {
    return thrust::raw_pointer_cast(v.data());
}

inline const double* raw_ptr(const thrust::device_vector<double>& v) {
    return thrust::raw_pointer_cast(v.data());
}

inline int* raw_ptr(thrust::device_vector<int>& v) {
    return thrust::raw_pointer_cast(v.data());
}

inline const int* raw_ptr(const thrust::device_vector<int>& v) {
    return thrust::raw_pointer_cast(v.data());
}

inline int to_int(std::size_t n, const char* what) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(what) + ": size exceeds INT_MAX");
    }
    return static_cast<int>(n);
}

}  // namespace

bool gpu_backend_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return false;
    }
    return count > 0;
}

CGResult solve_conjugate_gradient_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options,
    const std::vector<double>& x0) {
    if (!A.is_valid()) {
        throw std::invalid_argument("solve_conjugate_gradient_gpu: invalid matrix");
    }
    if (A.rows != A.cols) {
        throw std::invalid_argument("solve_conjugate_gradient_gpu: A must be square");
    }
    if (b.size() != static_cast<std::size_t>(A.rows)) {
        throw std::invalid_argument(
            "solve_conjugate_gradient_gpu: b size mismatch");
    }
    if (!x0.empty() && x0.size() != static_cast<std::size_t>(A.cols)) {
        throw std::invalid_argument(
            "solve_conjugate_gradient_gpu: x0 size mismatch");
    }

    if (!gpu_backend_available()) {
        throw std::runtime_error(
            "solve_conjugate_gradient_gpu: no CUDA-capable GPU detected");
    }

    const int n = to_int(static_cast<std::size_t>(A.rows), "A.rows");
    const int nnz = to_int(A.values.size(), "A.values.size()");

    CGResult result;
    result.x = x0.empty() ? std::vector<double>(A.cols, 0.0) : x0;

    thrust::device_vector<int> d_row_ptr(A.row_ptr.begin(), A.row_ptr.end());
    thrust::device_vector<int> d_col_idx(A.col_idx.begin(), A.col_idx.end());
    thrust::device_vector<double> d_values(A.values.begin(), A.values.end());
    thrust::device_vector<double> d_b(b.begin(), b.end());
    thrust::device_vector<double> d_x(result.x.begin(), result.x.end());
    thrust::device_vector<double> d_r(static_cast<std::size_t>(n), 0.0);
    thrust::device_vector<double> d_p(static_cast<std::size_t>(n), 0.0);
    thrust::device_vector<double> d_Ap(static_cast<std::size_t>(n), 0.0);

    CublasHandle cublas;
    CusparseHandle cusparse;
    CsrDescriptor A_desc(
        A.rows,
        A.cols,
        nnz,
        raw_ptr(d_row_ptr),
        raw_ptr(d_col_idx),
        raw_ptr(d_values));

    const std::size_t workspace_size = spmv_workspace_size(
        cusparse.get(),
        A_desc.get(),
        n,
        raw_ptr(d_x),
        raw_ptr(d_Ap));
    thrust::device_vector<char> workspace(workspace_size);

    spmv(
        cusparse.get(),
        A_desc.get(),
        n,
        raw_ptr(d_x),
        raw_ptr(d_Ap),
        workspace_size == 0 ? nullptr : workspace.data().get());

    copy(cublas.get(), n, raw_ptr(d_b), raw_ptr(d_r));
    axpy(cublas.get(), n, -1.0, raw_ptr(d_Ap), raw_ptr(d_r));
    copy(cublas.get(), n, raw_ptr(d_r), raw_ptr(d_p));

    double rs_old = dot(cublas.get(), n, raw_ptr(d_r), raw_ptr(d_r));
    result.residual_norm = std::sqrt(rs_old);
    if (result.residual_norm <= options.tolerance) {
        thrust::copy(d_x.begin(), d_x.end(), result.x.begin());
        result.converged = true;
        return result;
    }

    for (std::size_t k = 0; k < options.max_iterations; ++k) {
        spmv(
            cusparse.get(),
            A_desc.get(),
            n,
            raw_ptr(d_p),
            raw_ptr(d_Ap),
            workspace_size == 0 ? nullptr : workspace.data().get());

        const double denom = dot(cublas.get(), n, raw_ptr(d_p), raw_ptr(d_Ap));
        if (std::abs(denom) <= std::numeric_limits<double>::epsilon()) {
            result.iterations = k;
            result.residual_norm = std::sqrt(rs_old);
            result.converged = false;
            thrust::copy(d_x.begin(), d_x.end(), result.x.begin());
            return result;
        }

        const double alpha = rs_old / denom;
        axpy(cublas.get(), n, alpha, raw_ptr(d_p), raw_ptr(d_x));
        axpy(cublas.get(), n, -alpha, raw_ptr(d_Ap), raw_ptr(d_r));

        const double rs_new = dot(cublas.get(), n, raw_ptr(d_r), raw_ptr(d_r));
        result.iterations = k + 1;
        result.residual_norm = std::sqrt(rs_new);
        if (result.residual_norm <= options.tolerance) {
            result.converged = true;
            thrust::copy(d_x.begin(), d_x.end(), result.x.begin());
            return result;
        }

        const double beta = rs_new / rs_old;
        scale(cublas.get(), n, beta, raw_ptr(d_p));
        axpy(cublas.get(), n, 1.0, raw_ptr(d_r), raw_ptr(d_p));
        rs_old = rs_new;
    }

    thrust::copy(d_x.begin(), d_x.end(), result.x.begin());
    result.converged = false;
    return result;
}

}  // namespace cg
