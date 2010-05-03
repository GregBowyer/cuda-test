CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE_DIRECTORIES(../common/inc)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# Anything other than -D or /D is not passed along to nvcc.
add_definitions(-DMULTIPLIER=2)

# You can set BUILD_SHARED_LIBS or you can pass STATIC, SHARED, or MODULE to
# CUDA_ADD_LIBRARY.  Remember that BUILD_SHARED_LIBS is only respected by
# CUDA_ADD_LIBRARY.  If you use CUDA_COMPILE or CUDA_WRAP_SRCS you must specify
# SHARED or STATIC as a parameter.
#  -arch sm_13
set(BUILD_SHARED_LIBS ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CMAKE_C_FLAGS_DEBUG "-fno-omit-frame-pointer -g -gdwarf-2")

# GetOpts handler used to pull command line opts from the OS
add_library(support_code
    STATIC
    opts.c
    timer.cc
)
TARGET_LINK_LIBRARIES(support_code cuda-utils)

# KeywordMatrix lib creates a keyword/token matrix from a list of keywords
add_library(KeywordMatrix
    STATIC
    KeywordMatrix.cpp
    KeywordMatrix.h
)
TARGET_LINK_LIBRARIES(KeywordMatrix support_code)

CUDA_ADD_LIBRARY(Covariance
    STATIC
    covariance.h
    covariance.cu
    OPTIONS ""
    RELEASE --use_fast_math -DNDEBUG
    DEBUG -g -DDEBUG
)

# Then link the shared library to another executable.
ADD_EXECUTABLE(cuda-test
  main.cc
)

# Specify the dependency.
TARGET_LINK_LIBRARIES(cuda-test
  cuda-utils
  KeywordMatrix
  Covariance
  support_code
)
