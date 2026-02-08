cmake_minimum_required(VERSION 3.16)
project(cg_solver VERSION 0.1.0 LANGUAGES CXX)

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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cgsolver PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cg_driver apps/cg_driver.cpp)
target_link_libraries(cg_driver PRIVATE cgsolver)
