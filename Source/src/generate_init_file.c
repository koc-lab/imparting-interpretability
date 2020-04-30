#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Additional headers
#include <time.h>
#include "helperfuncs.h"

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

int verbose = 2; // 0, 1, or 2
int vector_size = 50; // Word vector size
real *W;
long long vocab_size;
char *vocab_file;

// Additional files
char *init_file; // Initialization file. One file per corpus. Required.
//


/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
	long long a, b;
	FILE* finit;
	vector_size++; // Temporarily increment to allocate space for bias
    
	/* Allocate space for word vectors and context word vectors */
	a = posix_memalign((void **)&W, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
	// Initialization file
	finit = fopen(init_file,"wb");
	// Random-init
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
	// Write to file
	fwrite(W, sizeof(real), 2 * (long long)vocab_size * vector_size, finit);
	fclose(finit);
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\nExample usage:\n");
        printf("./generate_init_file -vocab-file vocab.txt -verbose 2 -vector-size 100\n\n");
        return 0;
    }
    
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else strcpy(vocab_file, (char *)"vocab.txt");
    
    // Additional input arguments defined here: (declare such variables globally with a default definition)
    //
    {
    	init_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    	if ((i = find_arg((char *)"-INIT_FILE", argc, argv)) > 0) strcpy(init_file, argv[i + 1]);
        else strcpy(init_file, (char *)"out/init.bin");
    }


    vocab_size = 0;
    fid = fopen(vocab_file, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
    fclose(fid);
    
    if(verbose > 1) fprintf(stderr,"Initializing parameters...\n");
    initialize_parameters();
    return 0;
}
