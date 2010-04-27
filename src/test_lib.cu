#include <external_dependency.h>
#include <iostream>

#if test_lib_EXPORTS
#  if defined( _WIN32 ) || defined( _WIN64 )
#    define TEST_LIB_API __declspec(dllexport)
#  else
#    define TEST_LIB_API
#  endif
#else
#  define TEST_LIB_API
#endif

#define BLOCK_SIZE 16

__device__ inline int get_intersections(Vector *itrs, int t1, int t2) {
	int n = 0;
	Vector k1 = itrs[t1];
	Vector k2 = itrs[t2];
	for (int i = 0; i < k1.size; i++) {
		int *v1 = k1.values;
		for (int j = 0; j < k2.size; j++) {
			int *v2 = k2.values;
			if (v1[i] == v2[j])
				n++;
		}
	}

	return n;
}

__global__ void calc(float** A, int* tokens, Vector* intersections, int wT, int wK) {

	int i = threadIdx.x;
	int j = threadIdx.y;
	
	
	float t1 = tokens[i];
	float t2 = tokens[j];
	float v = 0;
	if (i > 0 && j < i) {
		// already calculated
	} else {
		float t01 = (1 - t1) / wK;
		float t00 = -t1 / wK;
		if (i == j) {
			// calculate diagonal
			v = ((__powf(t01, 2) * t1) + ((__powf(t00, 2) * (wK - t1)))) / wK;
		} else {
			float nn = (float) get_intersections(intersections, i, j);
			//float nn = 1;
			float t10 = __fdividef(-t2, wK);
			float t11 = __fdividef((1 - t2), wK);
			v = ((nn * t01 * t11) + ((t1 - nn) * t01 * t10) + ((t2 - nn) * t00 * t11) + ((wK - (t2 + t1 - nn)) * t00 * t10)) / wK;
		}
	}
	//if (v != 0)
	//	std::cout << i << ", " << j << ", " << v << std::endl;
	// A(i, j) = v;
	A[i][j] = v;
}

TEST_LIB_API int covariance(float** h_A, int* tokens, Vector* intersections, int wT, int wK) {

	//cudaFree(0);
	//CHECK_CUDA_ERROR();

	Size mem_size_T = sizeof(int) * wT;
	int* d_Tokens;
	cudaMalloc((void**) &d_Tokens, mem_size_T);
	
	Size mem_size_I = sizeof(Vector) * wT;
	Vector* d_Intersections;
	cudaMalloc((void**) &d_Intersections, mem_size_I);
	
	cudaMemcpy(d_Tokens, tokens, mem_size_T, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR();
	cudaMemcpy(d_Intersections, intersections, mem_size_I, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR();
	
	// allocate device memory for result
	Size size_A = wT * wT;
	Size mem_size_A = sizeof(float) * size_A;
	float** d_A;
	cudaMalloc((void**) &d_A, mem_size_A);
	CHECK_CUDA_ERROR();
	
    //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 grid(wT / threads.x, wT / threads.y);
	
	int numBlocks = 1;
	dim3 threadsPerBlock(3,3);
	
	calc<<<numBlocks, threadsPerBlock>>>(d_A, d_Tokens, d_Intersections, wT, wK);
	cudaThreadSynchronize();

	
	//for (int i = 0; i < 10; i++) {
	//	printf("%u %f\n", i, d_A[i]);
	//}
	
	cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR();
	
	cudaFree(d_Tokens);
	cudaFree(d_Intersections);
	cudaFree(d_A);
	
	return 0;
}
/*
 	//for (int i = 0; i < wT; i++) {
		float t1 = tokens[i];
		//for (int j = 0; j < wT; j++) {
			float t2 = tokens[j];
			float v = 0;
			if (i > 0 && j < i) {
				// already calculated
			} else {
				if (i == j) {
					// calculate diagonal
					v = ((pow((1 - t1 / wK), 2) * t1) + ((pow((-t1 / wK), 2) * (wK - t1)))) / wK;
				} else {
					//float nn = (float) get_intersections(intersections, i, j);
					float nn = 1;
					float t00 = -t1 / wK;
					float t01 = 1 - t1 / wK;
					float t10 = -t2 / wK;
					float t11 = 1 - t2 / wK;
					v = ((nn * t01 * t11) + ((t1 - nn) * t01 * t10) + ((t2 - nn) * t00 * t11) + ((wK - (t2 + t1 - nn)) * t00 * t10)) / wK;
				}
			}
			//if (v != 0)
			//	std::cout << i << ", " << j << ", " << v << std::endl;
			// A(i, j) = v;
			A[i * j] = 1;
		//}
	//}
	 */
 

