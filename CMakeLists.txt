cmake_minimum_required(VERSION 3.16)
project(cg_solver VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cgsolver
    src/cg.cpp
    src/sparse_csr.cpp
)

target_include_directories(cgsolver
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    find_path(CG_CUBLAS_INCLUDE
        NAMES cublas_v2.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIR}
    )
    find_path(CG_CUSPARSE_INCLUDE
        NAMES cusparse.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIR}
    )
    find_library(CG_CUBLAS_LIBRARY
        NAMES cublas
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64 ${CUDAToolkit_TARGET_DIR}/lib
    )
    find_library(CG_CUSPARSE_LIBRARY
        NAMES cusparse
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64 ${CUDAToolkit_TARGET_DIR}/lib
    )

    if(CG_CUBLAS_INCLUDE AND CG_CUSPARSE_INCLUDE AND CG_CUBLAS_LIBRARY AND CG_CUSPARSE_LIBRARY)
        target_sources(cgsolver PRIVATE src/cg_gpu.cu)
        set_source_files_properties(src/cg_gpu.cu PROPERTIES LANGUAGE CUDA)
        set_property(TARGET cgsolver PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        set_target_properties(cgsolver PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_link_libraries(cgsolver PUBLIC CUDA::cudart ${CG_CUBLAS_LIBRARY} ${CG_CUSPARSE_LIBRARY})
        target_compile_definitions(cgsolver PUBLIC CGSOLVER_USE_CUDA=1)
    else()
        message(WARNING "CUDA compiler found but cuBLAS/cuSPARSE headers or libraries are missing; building GPU stub.")
        target_sources(cgsolver PRIVATE src/cg_gpu_stub.cpp)
    endif()
else()
    target_sources(cgsolver PRIVATE src/cg_gpu_stub.cpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cgsolver PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

add_executable(cg_driver apps/cg_driver.cpp)
target_link_libraries(cg_driver PRIVATE cgsolver)
