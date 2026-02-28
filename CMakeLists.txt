cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES C CXX CUDA)

# C++/CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# tokenizers-cpp
set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/tokenizers-cpp)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_executable(main main.cpp kernel.cu manager.cpp)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120"
)

target_include_directories(main PRIVATE ${TOKENIZER_CPP_PATH}/include)
target_link_libraries(main PRIVATE tokenizers_cpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
