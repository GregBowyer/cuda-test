#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "external_dependency.h"

using namespace std;

extern int doit();

int main(int argc, char **argv) {

	//cout << "hello dude" << endl;






	CHECK_CUDA_ERROR();
	return doit();
}
;

