#ifndef OPTS_H
#define OPTS_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

struct options {
    int verbosity;
    char* input_file;
    char* output_file;
};

extern int process_commandline_options (struct options *opts, int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif
