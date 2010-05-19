#ifndef _COVARIANCE_KERNEL_H_
#define _COVARIANCE_KERNEL_H_

#define A(i, j) result[i + (j * wT)]
#define B(i, j) v[i + (j * wT)]

/*
__device__ float get_intersections(int* intr, int t1, int t2, int wI) {
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

	return (float) n;
}
*/

__global__ void calc(float* result, int* tokens, int* intr, int wT, int wK, int wI) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float t1 = (float) tokens[i];
	float t2 = (float) tokens[j];
	float v = 0;
	if (i >= j) {
		float t00 = -t1 / wK;
		float t01 = 1 - t1 / wK;
		if (i == j) {
			// calculate diagonal
			v = ((t01 * t01 * t1) + (t00 * t00 * (wK - t1))) / wK;
		} else {
			//float nn = get_intersections(intr, i, j, wI);
			float nn = (float) intr[j + i * wT];
			float t10 = -t2 / wK;
			float t11 = 1 - t2 / wK;
			v = ((nn * t01 * t11) + ((t1 - nn) * t01 * t10) + ((t2 - nn) * t00 * t11) + ((wK - (t2 + t1 - nn)) * t00 * t10)) / wK;
		}
		
		//__syncthreads();
		result[i + (j * wT)] = v;
		result[j + (i * wT)] = v;
		//A(i, j) = B(i, j);
		//A(j, i) = B(i, j);
	}
}

#endif // #ifndef _COVARIANCE_KERNEL_H_
