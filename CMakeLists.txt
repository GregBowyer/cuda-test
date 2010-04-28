# The name of our project is "HELLO". CMakeLists files in this project can 
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and 
# to the root binary directory of the project as ${HELLO_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.6) 
project (Cuda) 

include(ExternalProject)

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

ExternalProject_Add(breakpad-lib
    SVN_REPOSITORY http://google-breakpad.googlecode.com/svn/trunk/
    # We need to patchup the autotools chain a little, automake 1.11.1
    # and libtool 2.6.9 are _really_ new and the code builds fine without
    PATCH_COMMAND /usr/bin/libtoolize --force --copy &&
        sed -i s/AM_INIT_AUTOMAKE\(subdir-objects\ tar-ustar\ 1.11.1/AM_INIT_AUTOMAKE\(subdir-objects\ tar-ustar\ 1.11/ ./configure.ac
    CONFIGURE_COMMAND ./configure
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make all
    # No install 
    INSTALL_COMMAND true
)

subdirs(
    src
)

