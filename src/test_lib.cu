
#include <external_dependency.h>
#include <iostream>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

__global__ void times2_kernel(int *in, int *out) {

  for (unsigned int i=0;i<blockDim.x;++i) {
    // /*const*/ unsigned int thread = threadIdx.x;
    out[threadIdx.x] = in[threadIdx.x] * MULTIPLIER;
  }
};

void times2(int* in, int* out, int dim) {
  // Setup kernel problem size
  dim3 blocksize(dim,1,1);
  dim3 gridsize(1,1,1);

  // Call kernel
  times2_kernel<<<gridsize, blocksize>>>(in, out);
}

#if test_lib_EXPORTS
#  if defined( _WIN32 ) || defined( _WIN64 )
#    define TEST_LIB_API __declspec(dllexport)
#  else
#    define TEST_LIB_API
#  endif
#else
#  define TEST_LIB_API
#endif

TEST_LIB_API int doit()
{
	
	cusp::coo_matrix<int, float, cusp::device_memory> B;
	cusp::io::read_matrix_market_file(B, "/home/amerlo/workspace/cuda-test/matrix.mm");
	cusp::print_matrix(B);

  return 0;
}


