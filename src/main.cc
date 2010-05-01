#include "opts.h"
#include "KeywordMatrix.h"

using namespace std;

extern int covariance(map<string, int> tokens, map<string, set<int> > intersections, int wK);

int main(int argc, char **argv) {

    options prog_opts;
    process_commandline_options(&prog_opts, argc, argv);

	string input_file = prog_opts.input_file;
    printf("Input file: %s\n", prog_opts.input_file);
	string output_file = prog_opts.output_file;

	KeywordMatrix km(input_file.c_str());

	covariance(km.get_tokens(), km.get_intersections(), km.num_keywords());

	return 0;
}
