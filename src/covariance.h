#ifndef EXTERNDEPENDENCY__H
#define EXTERNDEPENDENCY__H

typedef unsigned int Size;

struct Vector {
	int size;
	int* values;
};

#ifdef __CUDACC__
#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
        cudaError err = call;                                                \
        if( cudaSuccess != err) {                                            \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                        __FILE__, __LINE__, cudaGetErrorString( err) );      \
                exit(EXIT_FAILURE);                                          \
        } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
        CUDA_SAFE_CALL_NO_SYNC(call);                                        \
        cudaError err = cudaThreadSynchronize();                             \
        if( cudaSuccess != err) {                                            \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                        __FILE__, __LINE__, cudaGetErrorString( err) );      \
                exit(EXIT_FAILURE);                                          \
        } } while (0)

#  define CUDA_CHECK_ERROR() do {                                            \
        cudaError err = cudaGetLastError();                                  \
        if( cudaSuccess != err) {                                            \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                        __FILE__, __LINE__, cudaGetErrorString( err) );      \
                exit(EXIT_FAILURE);                                          \
        } } while (0)

#else
// drop all CUDA calls
#define CUDA_SAFE_CALL_NO_SYNC(x)
#define CUDA_SAFE_CALL(x)
#endif

#define CHECK_CUDA_ERROR() \
  { \
    cudaThreadSynchronize(); \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
      printf("error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      return 1; \
    } \
  }
#endif