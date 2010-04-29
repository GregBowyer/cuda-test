#include <iostream>
#include <map>
#include <set>

#include "covariance.h"
#include "covariance_kernel.cu"

using namespace std;

#define BLOCK_SIZE 16

int covariance(map<string, int> tokens, map<string, set<int> > intersections, int wK) {
	cudaFree(0);
	CHECK_CUDA_ERROR();

	int wT = tokens.size();
	Size mem_size_T = sizeof(int) * wT;
	int* h_Tokens = (int*) malloc(mem_size_T);

	// map token info to c array and copy to device
	int index = 0; // temp counter
	for (std::map<std::string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		h_Tokens[index++] = (*it).second;
	}

	int* d_Tokens;
	cudaMalloc((void**) &d_Tokens, mem_size_T);
	cudaMemcpy(d_Tokens, h_Tokens, mem_size_T, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR();

	// map intersections info to c array
	int wI = 0;
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		int s = ((*it).second).size();
		if (s > wI)
			wI = s;
	}

	index = 0;
	Size mem_size_I = sizeof(int) * wT * wI;
	int* h_Intr = (int*) malloc(mem_size_I);
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		set<int> tokenSet = (*it).second;
		for (set<int>::iterator itt = tokenSet.begin(); itt != tokenSet.end(); itt++) {
			h_Intr[index++] = *itt;
		}
		// pad with zeros
		if (tokenSet.size() < wI) {
			for (int i = 0; i < wI - tokenSet.size(); i++)
				h_Intr[index++] = 0;
		}
	}
	
	int* d_Intr;
	cudaMalloc((void**) &d_Intr, mem_size_I);
	cudaMemcpy(d_Intr, h_Intr, mem_size_I, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR();

	// allocate memory for the result
	Size mem_size_result = sizeof(Real) * wT * wT;
	Real* h_result = (Real*) malloc(mem_size_result);
	Real* d_result;
	memset(h_result, 0, mem_size_result);
	cudaMalloc((void **) &d_result, mem_size_result);
	cudaMemset(d_result, 0, mem_size_result);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(wT / threadsPerBlock.x, wT / threadsPerBlock.y);

	calc<<<numBlocks, threadsPerBlock>>>(d_result, d_Tokens, d_Intr, wT, wK, wI);
	cudaThreadSynchronize();

	cudaMemcpy(h_result, d_result, mem_size_result, cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR();
	
	for (int i = 0; i < (wT * wT); i++) {
		printf("%u %f\n", i, h_result[i]);
	}

	cudaFree(d_Tokens);
	cudaFree(d_Intr);
	cudaFree(d_result);
	delete h_Tokens;
	delete h_Intr;
	delete h_result;

	return 0;
}

