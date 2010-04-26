
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

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
set(BUILD_SHARED_LIBS ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
CUDA_ADD_LIBRARY(test_lib
  test_lib.cu
  external_dependency.h
  # SHARED
  # STATIC
  OPTIONS "-DSTUFF=blah blah"
  RELEASE --use_fast_math -DNDEBUG
  DEBUG -g -DDEBUG
  )

# Using the CUDA_COMPILE macro
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(source_files test_bin.cu)
CUDA_COMPILE(CUDA_FILES test_bin.cu)

ADD_EXECUTABLE(cuda-test
  ${CUDA_FILES}
  ${source_files}
  main.cc
  external_dependency.h
  keyword_util.h
  )
TARGET_LINK_LIBRARIES(cuda-test
 ${CUDA_LIBRARIES}
 )

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

