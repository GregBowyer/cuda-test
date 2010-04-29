# The name of our project is "HELLO". CMakeLists files in this project can 
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and 
# to the root binary directory of the project as ${HELLO_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.6) 
project (cuda-test) 

find_package(Cuda QUIET REQUIRED)
find_package(Boost QUIET REQUIRED)

#set(CMAKE_BUILD_TYPE "Debug")
set(CUDA_VERBOSE_BUILD ON)

enable_testing()

# there are problems compiling 64-bit on mac osx.
# this compile everything in 32-bit
if (CMAKE_SYSTEM_NAME MATCHES Darwin)
    add_definitions(-Wall -m32)
    set(CUDA_64_BIT_DEVICE_CODE OFF)
    set(CMAKE_MODULE_LINKER_FLAGS -m32)
    set(CMAKE_C_FLAGS -m32) 
    set(CMAKE_CXX_FLAGS -m32)
endif()

add_subdirectory( src tests )
