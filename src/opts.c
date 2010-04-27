#include "opts.h"

static const char *opt_string = "ov:h?";

void print_help(char* prog_name, int exit_code) {
    printf("%s basic cuda toy covarience example\n", prog_name);
    printf("usage: %s [-options] [file ...] - Read the given file as the input matrix\n", prog_name);
    printf("Arguments:\n");
    printf("\t-h\t\tprint this help and exit\n");
    printf("\t-o\t\tuse a different output file\n");
    exit(exit_code);
}

extern int process_commandline_options(struct options* opts, int argc, char *argv[]) {
    int opt;

    opts->output_file = "/tmp/matrix.mm";
    opts->verbosity = 0;

    char* prog_name = argv[0];

    if(argc == 1) {
        fprintf(stderr, "We require the name of the input matrix\n");
        print_help(prog_name, 1);
    } else {

        while((opt = getopt(argc, argv, opt_string)) != -1) {
            switch(opt) {
                case 'h':
                case '?':
                    print_help(prog_name, 0);
                    break;
                case 'o':
                    opts->output_file = optarg;
                    break;
                case 'v':
                    opts->verbosity++;
                    break;
                default:
                    break;
            }
        }

        opts->input_file = argv[optind];
    }

   return 0;
}
