
#ifndef _COVARIANCE_KERNEL_H_
#define _COVARIANCE_KERNEL_H_

#define Real float
//#define Real double

__device__ Real get_intersections(int* intr, int t1, int t2, int wI) {
	int n = 0;
	for (int i = 0; i < wI; i++) {
		int x1 = (t1 * wI) + i;
		if (intr[x1] == 0)
			break;
		
		for (int j = 0; j < wI; j++) {
			int x2 = (t2 * wI) + j;
			if (intr[x2] == 0)
				break;			
			
			if (intr[x1] == intr[x2])
				n++;
		}
	}

	return (Real) n;
}

__global__ void calc(Real* result, int* tokens, int* intr, int wT, int wK, int wI) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	Real t1 = (Real) tokens[i];
	Real t2 = (Real) tokens[j];
	float v = 0;
	if (i >= j) {
		Real t00 = -t1 / wK;
		Real t01 = 1 - t1 / wK;
		if (i == j) {
			// calculate diagonal
			v = ((t01 * t01 * t1) + (t00 * t00 * (wK - t1))) / wK;
		} else {
			Real nn = get_intersections(intr, i, j, wI);
			Real t10 = -t2 / wK;
			Real t11 = 1 - t2 / wK;
			v = ((nn * t01 * t11) + ((t1 - nn) * t01 * t10) + ((t2 - nn) * t00 * t11) + ((wK - (t2 + t1 - nn)) * t00 * t10)) / wK;
		}
		result[i + (j * wT)] = v;
		result[j + (i * wT)] = v;
	}
}

#endif // #ifndef _COVARIANCE_KERNEL_H_