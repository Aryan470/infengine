cmake_minimum_required(VERSION 3.18)
project(InfEngine LANGUAGES C CXX CUDA)

# C++/CUDA standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# tokenizers-cpp
set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/tokenizers-cpp)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

file(GLOB MAIN_SOURCES
    ${CMAKE_SOURCE_DIR}/kernels/*.cu
    ${CMAKE_SOURCE_DIR}/kernels/*.cuh
    ${CMAKE_SOURCE_DIR}/kernels/*.cpp
    ${CMAKE_SOURCE_DIR}/kernels/*.h
    ${CMAKE_SOURCE_DIR}/*.cpp
    ${CMAKE_SOURCE_DIR}/*.cu
    ${CMAKE_SOURCE_DIR}/*.h
    ${CMAKE_SOURCE_DIR}/*.cuh
)

add_executable(main ${MAIN_SOURCES})

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120"
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(main PRIVATE ${TOKENIZER_CPP_PATH}/include)
target_link_libraries(main PRIVATE tokenizers_cpp CUDA::cublas)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/tests/*.cu
    ${CMAKE_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_SOURCE_DIR}/kernels/*.cu
    ${CMAKE_SOURCE_DIR}/kernels/*.cpp
)

add_executable(test_kernels ${TEST_SOURCES})

target_link_libraries(test_kernels GTest::gtest_main CUDA::cublas ${CUDA_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(test_kernels)