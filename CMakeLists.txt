# The name of our project is "HELLO". CMakeLists files in this project can 
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and 
# to the root binary directory of the project as ${HELLO_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.6) 
project (Cuda) 

find_package(CUDA QUIET REQUIRED)

# there are problems compiling 64-bit on mac osx.
# this compile everything in 32-bit
if (CMAKE_SYSTEM_NAME MATCHES Darwin)
    add_definitions(-Wall -m32)
    set(CUDA_64_BIT_DEVICE_CODE OFF)
    set(CMAKE_MODULE_LINKER_FLAGS -m32)
    set(CMAKE_C_FLAGS -m32) 
    set(CMAKE_CXX_FLAGS -m32)
endif(CMAKE_SYSTEM_NAME MATCHES Darwin)

subdirs(
    src
)

