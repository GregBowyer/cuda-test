#include "opts.h"
#include "timer.h"
#include "KeywordMatrix.h"

using namespace std;

extern void covariance(float* result, map<string, int> tokens, map<string, set<int> > intersections, int wK);

int main(int argc, char **argv) {

	options prog_opts;
	process_commandline_options(&prog_opts, argc, argv);

	string input_file = prog_opts.input_file;
	printf("Input file: %s\n", prog_opts.input_file);
	string output_file = prog_opts.output_file;

	CUTimer *total_time = start_timing("Total Runtime");
	CUTimer *timer_load = start_timing("File Load");

	KeywordMatrix km(input_file.c_str());
	int wT = km.num_tokens();

	finish_timing(timer_load);

	unsigned int mem_size_result = sizeof(float) * wT * wT;
	float* result = (float*) calloc(1, mem_size_result);

	covariance(result, km.get_tokens(), km.get_intersections(), km.num_keywords());

	if(prog_opts.verbosity != 0) {
		for (int i = 0; i < (wT * wT); i++) {
			printf("%u %f\n", i, result[i]);
		}
	}

	finish_timing(total_time);

	free(result);
	return 0;
}
