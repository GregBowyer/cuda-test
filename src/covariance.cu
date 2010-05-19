#include <iostream>
#include <map>
#include <set>

#include "covariance.h"
#include "covariance_kernel.cu"
#include "timer.h"

using namespace std;

#define BLOCK_SIZE 16

int count_intr(int* intr, int t1, int t2, int wI) {
	int n = 0;
	for (int i = 0; i < wI; i++) {
		int x1 = (t1 * wI) + i;
		if (intr[x1] == -1)
			break;
		
		for (int j = 0; j < wI; j++) {
			int x2 = (t2 * wI) + j;
			if (intr[x2] == -1)
				break;			
			
			if (intr[x1] == intr[x2])
				n++;
		}
	}

	return n;
}

void covariance(float* h_result, map<string, int> tokens, map<string, set<int> > intersections, int wK) {
	Cuda_SAFE_CALL(cudaFree(0));
	CUTimer *gpu_total_timer = start_timing("[GPU] Total Time");
	CUTimer *mem_timer = start_timing("[GPU] Host-Device Memory Load");

	int wT = tokens.size();
	size_t mem_size_T = sizeof(int) * wT;
	int* h_Tokens = (int*) malloc(mem_size_T);

	// map token info to c array and copy to device
	int index = 0; // temp counter
	for (std::map<std::string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		h_Tokens[index++] = (*it).second;
	}

	int* d_Tokens;
	Cuda_SAFE_CALL(cudaMalloc((void**) &d_Tokens, mem_size_T));
	Cuda_SAFE_CALL(cudaMemcpy(d_Tokens, h_Tokens, mem_size_T, cudaMemcpyHostToDevice));

	// map intersections info to c array
	int wI = 0;
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		int s = ((*it).second).size();
		if (s > wI)
			wI = s;
	}

	index = 0;
	size_t mem_size_intr = sizeof(int) * wT * wI;
	int* intr = (int*) malloc(mem_size_intr);
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		set<int> tokenSet = (*it).second;
		for (set<int>::iterator itt = tokenSet.begin(); itt != tokenSet.end(); itt++) {
			intr[index++] = *itt;
		}
		// pad with -1
		if (tokenSet.size() < wI) {
			for (int i = 0; i < wI - tokenSet.size(); i++)
				intr[index++] = -1;
		}
	}
	
	unsigned int sizeT = wT * wT;
	size_t mem_size_I = sizeof(int) * sizeT;
	int* h_Intr = (int*) calloc(sizeT, mem_size_intr);
	for (unsigned int i = 0; i<wT; i++) {
		for (unsigned int j = 0; j < wT; j++) {
			if (i != j && i < j) {
				h_Intr[i + j * wT] = count_intr(intr, i, j, wI);
			}
		}
	}
	
	int* d_Intr;
	Cuda_SAFE_CALL(cudaMalloc((void**) &d_Intr, mem_size_I));
	Cuda_SAFE_CALL(cudaMemcpy(d_Intr, h_Intr, mem_size_I, cudaMemcpyHostToDevice));

	// allocate memory for the result
	size_t mem_size_result = sizeof(float) * wT * wT;
	float* d_result;
	Cuda_SAFE_CALL(cudaMalloc((void **) &d_result, mem_size_result));
	Cuda_SAFE_CALL(cudaMemset(d_result, 0, mem_size_result));

	finish_timing(mem_timer);

	CUTimer *calculation_timer = start_timing("[GPU] Calculation on card");
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(wT / threadsPerBlock.x, wT / threadsPerBlock.y);

	calc<<<numBlocks, threadsPerBlock, mem_size_result>>>(d_result, d_Tokens, d_Intr, wT, wK, wI);
	cudaThreadSynchronize();
	Cuda_CHECK_ERROR();
	finish_timing(calculation_timer);

	CUTimer *backcopy_timer = start_timing("[GPU] Device-Host Memory Fetch");
	Cuda_SAFE_CALL(cudaMemcpy(h_result, d_result, mem_size_result, cudaMemcpyDeviceToHost));
	
	cudaFree(d_Tokens);
	cudaFree(d_Intr);
	cudaFree(d_result);
	delete h_Tokens;
	delete h_Intr;
	finish_timing(backcopy_timer);
	finish_timing(gpu_total_timer);
}

