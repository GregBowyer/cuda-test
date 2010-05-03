CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE_DIRECTORIES(./inc)
CUDA_ADD_LIBRARY(cuda-utils
    src/bank_checker.cpp
    src/cmd_arg_reader.cpp
    src/cutil.cpp
    src/stopwatch.cpp
    src/stopwatch_linux.cpp

    STATIC
    OPTIONS ""
    RELEASE --use_fast_math -DNDEBUG
    DEBUG -g -DDEBUG
)
