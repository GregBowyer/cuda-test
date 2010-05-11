#ifndef COVARIANCE__H
#define COVARIANCE__H

#define Size unsigned int

#ifdef __CUDACC__
#  define Cuda_SAFE_CALL_NO_SYNC( call) do {                                 \
        cudaError err = call;                                                \
        if( cudaSuccess != err) {                                            \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                        __FILE__, __LINE__, cudaGetErrorString( err) );      \
                exit(EXIT_FAILURE);                                          \
        } } while (0)

#  define Cuda_SAFE_CALL( call) do {                                         \
        Cuda_SAFE_CALL_NO_SYNC(call);                                        \
        cudaError err = cudaThreadSynchronize();                             \
        if( cudaSuccess != err) {                                            \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                        __FILE__, __LINE__, cudaGetErrorString( err) );      \
                exit(EXIT_FAILURE);                                          \
        } } while (0)

#  define Cuda_CHECK_ERROR() do {                                            \
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

#endif // #ifndef COVARIANCE__H
