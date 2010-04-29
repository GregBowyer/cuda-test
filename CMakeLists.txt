# Cuda-test project 
cmake_minimum_required (VERSION 2.6) 
project (cuda-test) 

set(Boost_USE_STATIC_LIBS ON)
set(CUDA_VERBOSE_BUILD ON)

find_package(Cuda QUIET REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS unit_test_framework) 

# there are problems compiling 64-bit on mac osx.
# this compile everything in 32-bit
if (CMAKE_SYSTEM_NAME MATCHES Darwin)
    add_definitions(-Wall -m32)
    set(CUDA_64_BIT_DEVICE_CODE OFF)
    set(CMAKE_MODULE_LINKER_FLAGS -m32)
    set(CMAKE_C_FLAGS -m32) 
    set(CMAKE_CXX_FLAGS -m32)
endif()

enable_testing()

add_subdirectory( src )
add_subdirectory( tests )

