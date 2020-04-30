//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// Additional headers
#include <time.h>
#include <ctype.h>
#include "helperfuncs.h"

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *gradsq, *cost;
long long num_lines, *lines_per_thread, vocab_size;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;


// Toggles used for debugging
int ignore_init_file = 0; // Set to 1 to randomly generate the initial parameter values instead.
int forcing_enabled = 1; // Setting to 0 disables dim force by setting numForcedDims to 0
//
// The following are required to be read from a file
int *forcedDims = 0; // 0..(vector_size-1).
int **wordIdsPerForcedDim; // frequency ranks
int **polaritiesPerForcedDim; // 1 or -1
real **kvalsPerForcedDim;
//
// The following are inferred from the files read
int numForcedDims;
int *numWordsPerForcedDim;
//
// Globally defined for convenience (want to see the string forms of the forced words at all times during debugging)
char ***wordStringsPerForcedDim;


// File names
char *init_file; // Initialization file. One file per corpus (generate with a large vector_size).
char *forced_dims_file;
char *forced_word_ids_file;
char *polarities_file, *k_vals_file;
//


/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
    real* W_temp;
    long long a, b;
    FILE* finit;
    vector_size++; // Temporarily increment to allocate space for bias
    
    /* Allocate space for word vectors and context word vectors, and correspodning gradsq */
    a = posix_memalign((void **)&W, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
    if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }

    if(!ignore_init_file){
        // Initialization file
        finit = fopen(init_file, "rb");
        if(finit == NULL) {fprintf(stderr, "Unable to open file %s.\n", init_file); exit(1);}
        // Read from file
        fread(W, sizeof(real), 2 * (long long)vocab_size * vector_size, finit);
        fclose(finit);
    }
    else{
        // Random-init
        for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }

    // Init for gradsq
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) gradsq[a * vector_size + b] = 1.0; // So initial value of eta is equal to initial learning rate
    vector_size--;
}

/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a;
    long long id = (long long) vid;
    CREC cr;
    FILE *fin;

    // Forced dims/pols/kvals for the word pair under consideration
    int w1_num_forced_dims;
    int* word1_forced_dims = (int*) malloc(sizeof(int) * numForcedDims);
    int* word1_forced_dim_pols = (int*) malloc(sizeof(int) * numForcedDims);
    real* word1_kvals =  (real*) malloc(sizeof(real) * numForcedDims);
    //
    int w2_num_forced_dims;
    int* word2_forced_dims = (int*) malloc(sizeof(int) * numForcedDims);
    int* word2_forced_dim_pols = (int*) malloc(sizeof(int) * numForcedDims);
    real* word2_kvals =  (real*) malloc(sizeof(real) * numForcedDims);
    //

    fin = fopen(input_file, "rb");
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;
    
    for(a = 0; a<lines_per_thread[id]; a++)
    {
        fread(&cr, sizeof(CREC), 1, fin);
        if(feof(fin)) break;

        // Set the forced dims/pols/kvals vectors for both words
        {
            int i,j;
            w1_num_forced_dims = 0; w2_num_forced_dims=0;
            memset(word1_forced_dims, 0, numForcedDims * sizeof(int));
            memset(word2_forced_dims, 0, numForcedDims * sizeof(int));
            for(i=0;i<numForcedDims;i++)
                for(j=0;j<numWordsPerForcedDim[i];j++){
                    if(wordIdsPerForcedDim[i][j] == cr.word1){
                        word1_forced_dims[w1_num_forced_dims] = forcedDims[i];
                        word1_forced_dim_pols[w1_num_forced_dims] = polaritiesPerForcedDim[i][j];
                        word1_kvals[w1_num_forced_dims] = kvalsPerForcedDim[i][j];
                        w1_num_forced_dims++;
                    }
                    if(wordIdsPerForcedDim[i][j] == cr.word2){
                        word2_forced_dims[w2_num_forced_dims] = forcedDims[i];
                        word2_forced_dim_pols[w2_num_forced_dims] = polaritiesPerForcedDim[i][j];
                        word2_kvals[w2_num_forced_dims] = kvalsPerForcedDim[i][j];
                        w2_num_forced_dims++;
                    }
                }
        }

        // Cost and gradient calculations
        {
            long long l1, l2;
            real k_w1, k_w2;
            real dotprod;
            real weight;

            // Positions of the two words in the W & gradsq structures
            l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
            l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words

            // Cost calculation
            {
                int i;
                real diff;
                real bias1, bias2;
                real cost_forced_term = 0.0;
                real val; int pol;
                real cost1_del, cost2_del;

                // Bias terms
                bias1 = W[l1 + vector_size];
                bias2 = W[l2 + vector_size];

                // The dot product of the first word vector and the second context vector
                dotprod = dot(W+l1, W+l2, vector_size);
                dotprod = dotprod + bias1 + bias2; // Add the biases

                // The difference between word-vector inner products and the log-cooccurrence
                diff = dotprod - log(cr.val);

                // The cost term due to the forced dimensions for the two words
                for(i=0; i<w1_num_forced_dims; i++){
                    k_w1 = word1_kvals[i];
                    val = *(W + l1 + word1_forced_dims[i]);
                    pol = word1_forced_dim_pols[i];
                    cost1_del = recipCost(val, pol, k_w1);
                    cost_forced_term += recipCost(val, pol, k_w1);
                }
                for(i=0; i<w2_num_forced_dims; i++){
                    k_w2 = word2_kvals[i];
                    val = *(W + l2 + word2_forced_dims[i]);
                    pol = word2_forced_dim_pols[i];
                    cost2_del = recipCost(val, pol, k_w2);
                    cost_forced_term += recipCost(val, pol, k_w2);
                }

                // The weight term for the squared-error cost
                weight = (cr.val > x_max) ? 1 : pow(cr.val / x_max, alpha);

                // Calculate the cost
                cost[id] += 0.5 * weight * (diff * diff + cost_forced_term);
            }

            // Adagrad updates
            {
                real temp = weight * (dotprod - log(cr.val));
                real gradient_w1, gradient_w2, gradient_b;
                int i;
                int matches_1 = 0, matches_2 = 0;
                real val; int pol;

                // Updates for each component:
                for(i=0; i<vector_size; i++)
                {
                    // Gradient for word 1 (actual)
                    gradient_w1 = temp * W[l2 + i];
                    // Gradient for word 2 (context)
                    gradient_w2 = temp * W[l1 + i];

                    // Gradient from the forced term for word 1
                    if(matches_1<w1_num_forced_dims && i == word1_forced_dims[matches_1]){
                        val = *(W + l1 + i);
                        pol = word1_forced_dim_pols[matches_1];
                        k_w1 = word1_kvals[matches_1];

                        gradient_w1 += weight*recipCostDer(val, pol, k_w1);
                        matches_1++;
                    }

                    // Gradient from the forced term for word 2
                    if(matches_2<w2_num_forced_dims && i == word2_forced_dims[matches_2]){
                        val = *(W + l2 + i);
                        pol = word2_forced_dim_pols[matches_2];
                        k_w2 = word2_kvals[matches_2];

                        gradient_w2 += weight*recipCostDer(val, pol, k_w2);
                        matches_2++;
                    }

                    // Update for word 1
                    W[l1 + i] -= (eta*gradient_w1) / sqrt(gradsq[l1 + i]);
                    gradsq[l1 + i] += eta*gradient_w1 * eta*gradient_w1;
                    // Update for word 2
                    W[l2 + i] -= (eta*gradient_w2) / sqrt(gradsq[l2 + i]);
                    gradsq[l2 + i] += eta*gradient_w2 * eta*gradient_w2;
                }

                // For the bias terms
                gradient_b = temp;

                W[l1 + vector_size] -= (eta * gradient_b) / sqrt(gradsq[l1 + vector_size]);
                gradsq[l1 + vector_size] += eta*gradient_b * eta*gradient_b;

                W[l2 + vector_size] -= (eta * gradient_b) / sqrt(gradsq[l2 + vector_size]);
                gradsq[l2 + vector_size] += eta*gradient_b * eta*gradient_b;
            }
        }
        
    }

    // Free up unused memory
    free(word1_forced_dims);
    free(word2_forced_dims);
    free(word1_forced_dim_pols);
    free(word2_forced_dim_pols);
    free(word1_kvals);
    free(word2_kvals);

    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params() {
    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH);
    FILE *fid, *fout, *fgs;
    
    if(use_binary > 0) { // Save parameters in binary file
        sprintf(output_file,"%s.bin",save_W_file);
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout);
        fclose(fout);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
            for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if(use_binary != 1) { // Save parameters in text file
        sprintf(output_file,"%s.txt",save_W_file);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
        }
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
            // input vocab cannot contain special <unk> keyword
            if(strcmp(word, "<unk>") == 0) return 1;
            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if(save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }

        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + 1), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + 1), sizeof(real));
            word = "<unk>";

            int num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for(a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for(b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if(save_gradsq > 0) fclose(fgs);
    }
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int b;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if(verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if(verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if(verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    
    // Print information on forced dims to console
    if(!forcing_enabled) fprintf(stderr, "Forcing disabled.\n");
    else{
        fprintf(stderr, "Parameters read from disk.\n");
        fprintf(stderr, "Number of forced dims = %d\n", numForcedDims);
        fprintf(stderr, "Forced dim(s) = %d", forcedDims[0]+1);
        for(b=1; b<numForcedDims; b++) fprintf(stderr, ", %d", forcedDims[b]+1);
        fprintf(stderr, "\n");
        fprintf(stderr, "Number of forced words:\n");
        fprintf(stderr, "Dim 1: %d", numWordsPerForcedDim[0]);
        for(b=1; b<numForcedDims; b++){
            fprintf(stderr, ", Dim %d:", b+1);
            fprintf(stderr, " %d", numWordsPerForcedDim[b]);
        }
        fprintf(stderr, "\n");
    }

    // Lock-free asynchronous SGD
    for(b = 0; b < num_iter; b++) {
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        fprintf(stderr,"iter: %03d, cost: %lf\n", b+1, total_cost/num_lines);
    }
    fprintf(stderr, "\n");

    // Free up unused memory after training
    {
        int j,k;
        if(forcing_enabled){
            free(forcedDims);
            for (j=0; j<numForcedDims; j++){
                free(wordIdsPerForcedDim[j]);
                free(polaritiesPerForcedDim[j]);
                free(kvalsPerForcedDim[j]);
            }
            free(wordIdsPerForcedDim);
            free(polaritiesPerForcedDim);
            free(kvalsPerForcedDim);
            for (j=0; j<numForcedDims; j++){
                for(k=0; k<numWordsPerForcedDim[j]; k++) free(wordStringsPerForcedDim[j][k]);
                free(wordStringsPerForcedDim[j]);
            }
            free(wordStringsPerForcedDim);
            free(numWordsPerForcedDim);
        }
    }
    return save_params();
}

int get_forced_dims(){
    // Forced dimensions
    {
        FILE *fid_forced_dims;

        size_t buffersize = 0; // updated by the getline call
        char *buffer = 0; // allocated and adjusted as needed by getline
        int line_length; // includes the terminating newline

        char *c;
        int dim_ind;
        int start, stop, dim = -1;

        fid_forced_dims = fopen(forced_dims_file, "r");
        if(fid_forced_dims == NULL) {fprintf(stderr, "Unable to open file %s.\n", forced_dims_file); return 1;}

        dim_ind = 0;
        while ((line_length = getline(&buffer, &buffersize, fid_forced_dims)) > 0){
            if(*buffer == '#') continue;
            if(*buffer == '\n') continue;
            if(strpbrk(buffer, ".") != NULL){fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}

            if((c = strpbrk(buffer, ":")) != NULL){
                if(c == buffer) start = 0; else{
                    if(isdigit(*buffer)) start = atoi(buffer); // want to avoid disallowing 0 entirely
                    else{fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}
                }
                if(*(c+1) == '\n') stop = vector_size; else stop = atoi(c+1);
                if(stop <= start){fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}
                if(stop > vector_size){fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}
                forcedDims = realloc(forcedDims, sizeof(int) * (dim_ind + stop-start));
                for(dim=start; dim<stop; dim++) forcedDims[dim_ind++] = dim;
            }
            else if((dim = atoi(buffer)) >= vector_size){fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}
            else if(!isdigit(*buffer)){fprintf(stderr, "Incompatible file: %s.\n", forced_dims_file); return 1;}
            else{
                forcedDims = realloc(forcedDims, sizeof(int) * (dim_ind + 1));
                forcedDims[dim_ind++] = dim;
            }
        }
        numForcedDims = dim_ind; // assuming no overlaps among the ranges in the input file
        fclose(fid_forced_dims);
    }


    // Forced words (freq ids) per forced dimension
    {
        FILE *fid_forced_ids;

        size_t buffersize = 0; // updated by the getline call
        char *buffer = 0; // allocated and adjusted as needed by getline
        char *tmp_buffer = 0;
        int line_length; // includes the terminating newline

        int line_ind, j;
        int num_tokens;
        char* token;

        fid_forced_ids = fopen(forced_word_ids_file, "r");
        if(fid_forced_ids == NULL) {fprintf(stderr, "Unable to open file %s.\n", forced_word_ids_file); return 1;}

        numWordsPerForcedDim = malloc(sizeof(int) * numForcedDims);

        // Allocate memory and read
        wordIdsPerForcedDim = (int**) malloc(sizeof(int*) * numForcedDims);

        line_ind = 0;
        while ((line_length = getline(&buffer, &buffersize, fid_forced_ids)) > 0){
            if(*buffer == '#') continue;
            if(*buffer == '\n') continue;
            if(strpbrk(buffer, ".") != NULL){fprintf(stderr, "Incompatible file: %s.\n", forced_word_ids_file); return 1;}
            if(line_ind == numForcedDims){fprintf(stderr, "Incompatible file: %s.\n", forced_word_ids_file); return 1;}

            // Get the number of tokens in the line
            num_tokens = 0;
            tmp_buffer = (char*) realloc(tmp_buffer, buffersize);
            strcpy(tmp_buffer, buffer);
            for(token = strtok(tmp_buffer, " "); token != NULL; token = strtok(NULL, " ")) num_tokens++;
            numWordsPerForcedDim[line_ind] = num_tokens;

            // Store the tokens
            wordIdsPerForcedDim[line_ind] = (int *) malloc(sizeof(int) * num_tokens);
            token = strtok(buffer, " ");
            for(j=0; j<num_tokens; j++){
                if(atoi(token) > vocab_size){fprintf(stderr, "Incompatible file: %s.\n", forced_word_ids_file); return 1;}
                if(atoi(token) <= 0){fprintf(stderr, "Incompatible file: %s.\n", forced_word_ids_file); return 1;}
                wordIdsPerForcedDim[line_ind][j] = atoi(token);
                token = strtok(NULL, " ");
            }

            // Next line
            line_ind++;
        }
        if(line_ind < numForcedDims){fprintf(stderr, "Incompatible file: %s.\n", forced_word_ids_file); return 1;}
        free(tmp_buffer);
        free(buffer);
        fclose(fid_forced_ids);
    }


    // Polarities
    {
        FILE *fid_polarities;

        size_t buffersize = 0; // updated by the getline call
        char *buffer = 0; // allocated and adjusted as needed by getline
        int line_length; // includes the terminating newline

        int line_ind, j, k, l;
        char* token;
        int md;

        // Polarities
        fid_polarities = fopen(polarities_file, "r");
        if(fid_polarities == NULL) {fprintf(stderr, "Unable to open file %s.\n", polarities_file); return 1;}

        // Allocate memory and read
        // polaritiesPerForcedDim = (int**) malloc(sizeof(int*) * numForcedDims);
        polaritiesPerForcedDim = (int**) calloc(numForcedDims, sizeof(int*)); // initialize to NULLs

        line_ind = 0;
        while ((line_length = getline(&buffer, &buffersize, fid_polarities)) > 0){
            if(*buffer == '#') continue;
            if(*buffer == '\n') continue;
            if(line_ind == numForcedDims){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
            if(strspn(buffer, "+- *") != line_length-1){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}

            // Store the tokens
            token = strtok(buffer, " ");
            for(j=0; j<numWordsPerForcedDim[line_ind]; j++){
                if((md = strspn(token, "*")) == 2){
                    if(j) {fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    if(line_ind) {fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    if(strpbrk(token+2, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    for(l=0; l<numForcedDims; l++){
                        polaritiesPerForcedDim[l] = (int *) malloc(sizeof(int) * numWordsPerForcedDim[l]);
                        for(k=0; k<numWordsPerForcedDim[l]; k++) {polaritiesPerForcedDim[l][k] = (*(token+2) == '+') ? 1:-1;}
                    }
                    token = strtok(NULL, " "); // should be NULL
                    do{line_length = getline(&buffer, &buffersize, fid_polarities);} while(line_length > 0 && (*buffer == '#' || *buffer == '\n')); // should be -1
                    break;
                }
                else if (md == 1) {
                    if(j) {fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    if(strpbrk(token+1, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    polaritiesPerForcedDim[line_ind] = (int *) malloc(sizeof(int) * numWordsPerForcedDim[line_ind]);
                    for(k=0; k<numWordsPerForcedDim[line_ind]; k++) {polaritiesPerForcedDim[line_ind][k] = (*(token+1) == '+') ? 1:-1;}
                    token = strtok(NULL, " "); // should be NULL
                    break;
                }
                else if (md == 0) {
                    if(token == NULL){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    if(strpbrk(token, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
                    if(polaritiesPerForcedDim[line_ind] == 0) {polaritiesPerForcedDim[line_ind] = (int *) malloc(sizeof(int) * numWordsPerForcedDim[line_ind]);}
                    polaritiesPerForcedDim[line_ind][j] = (*token == '+') ? 1:-1;
                    token = strtok(NULL, " ");
                }
                else{fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
            }
            if(token != NULL){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
            if(md == 2 && line_length > 0){fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}

            // Next line
            line_ind++;
        }
        if(line_ind < numForcedDims && md != 2) {fprintf(stderr, "Incompatible file: %s.\n", polarities_file); return 1;}
        free(buffer);
        fclose(fid_polarities);
    }


    // k-values
    {
        FILE *fid_k_vals;

        size_t buffersize = 0; // updated by the getline call
        char *buffer = 0; // allocated and adjusted as needed by getline
        int line_length; // includes the terminating newline

        int line_ind, j, k, l;
        char* token;
        int md;

        // k-values
        fid_k_vals = fopen(k_vals_file, "r");
        if(fid_k_vals == NULL) {fprintf(stderr, "Unable to open file %s.\n", k_vals_file); return 1;}

        // Allocate memory and read
        // kvalsPerForcedDim = (real**) malloc(sizeof(real*) * numForcedDims);
        kvalsPerForcedDim = (real**) calloc(numForcedDims, sizeof(real*)); // initialize to NULLs

        line_ind = 0;
        while ((line_length = getline(&buffer, &buffersize, fid_k_vals)) > 0){
            if(*buffer == '#') continue;
            if(*buffer == '\n') continue;
            if(line_ind == numForcedDims){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}

            // Store the tokens
            token = strtok(buffer, " ");
            for(j=0; j<numWordsPerForcedDim[line_ind]; j++){
                if((md = strspn(token, "*")) == 2){
                    if(j) {fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(line_ind) {fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(strpbrk(token+2, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(!isdigit(*(token+2))){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;} // want to avoid disallowing 0 entirely
                    for(l=0; l<numForcedDims; l++){
                        kvalsPerForcedDim[l] = (real *) malloc(sizeof(real) * numWordsPerForcedDim[l]);
                        for(k=0; k<numWordsPerForcedDim[l]; k++) {kvalsPerForcedDim[l][k] = atof(token+2);}
                    }
                    token = strtok(NULL, " "); // should be NULL
                    do{line_length = getline(&buffer, &buffersize, fid_k_vals);} while(line_length > 0 && (*buffer == '#' || *buffer == '\n')); // should be -1
                    break;
                }
                else if (md == 1) {
                    if(j) {fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(strpbrk(token+1, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(!isdigit(*(token+1))){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;} // want to avoid disallowing 0 entirely
                    kvalsPerForcedDim[line_ind] = (real *) malloc(sizeof(real) * numWordsPerForcedDim[line_ind]);
                    for(k=0; k<numWordsPerForcedDim[line_ind]; k++) {kvalsPerForcedDim[line_ind][k] = atof(token+1);}
                    token = strtok(NULL, " "); // should be NULL
                    break;
                }
                else if (md == 0) {
                    if(token == NULL){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(strpbrk(token, "*") != NULL){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
                    if(!isdigit(*token)){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;} // want to avoid disallowing 0 entirely
                    if(kvalsPerForcedDim[line_ind] == 0) {kvalsPerForcedDim[line_ind] = (real *) malloc(sizeof(real) * numWordsPerForcedDim[line_ind]);}
                    kvalsPerForcedDim[line_ind][j] = atof(token);
                    token = strtok(NULL, " ");
                }
                else{fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
            }
            if(token != NULL){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
            if(md == 2 && line_length > 0){fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}

            // Next line
            line_ind++;
        }
        if(line_ind < numForcedDims && md != 2) {fprintf(stderr, "Incompatible file: %s.\n", k_vals_file); return 1;}
        free(buffer);
        fclose(fid_k_vals);
    }


    // String forms (for ease with debugging)
    {
        FILE *fid;
        int matches;
        int target;
        int freqInd;
        char format[20], str[MAX_STRING_LENGTH + 1];
        long long unicount;
        int i,j,k;

        // Allocate memory
        wordStringsPerForcedDim = (char***) malloc(sizeof(char**) * numForcedDims);
        for (j = 0; j < numForcedDims; j++){
            wordStringsPerForcedDim[j] = (char**) malloc(sizeof(char*) * numWordsPerForcedDim[j]);
            for(k = 0; k < numWordsPerForcedDim[j]; k++)
                wordStringsPerForcedDim[j][k] = (char*) malloc(sizeof(char) * (MAX_STRING_LENGTH + 1));
        }

        // Read the vocab file and match words against their freqIds (requires that the words have freqId's in ascending order.)
        sprintf(format,"%%%ds %%lld", MAX_STRING_LENGTH);
        for(i=0;i<numForcedDims;i++){
            fid = fopen(vocab_file,"r");
            if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
            freqInd = 1;
            target = wordIdsPerForcedDim[i][0];
            matches = 0;
            while (fscanf(fid, format, str, &unicount) != EOF && matches<numWordsPerForcedDim[i]){
                while(freqInd == target){
                    strcpy(wordStringsPerForcedDim[i][matches], str);
                    matches++;
                    target = wordIdsPerForcedDim[i][matches];
                }
                freqInd++;
            }
            fclose(fid);
        }
    }
    return train_glove();
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
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        return 0;
    }
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    cost = malloc(sizeof(real) * num_threads);
    if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if(model != 0 && model != 1) model = 2;
    if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else strcpy(vocab_file, (char *)"vocab.txt");
    if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
    else strcpy(save_W_file, (char *)"vectors");
    if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
        strcpy(save_gradsq_file, argv[i + 1]);
        save_gradsq = 1;
    }
    else if(save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
    if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    else strcpy(input_file, (char *)"cooccurrence.shuf.bin");
    
    // Additional input arguments defined here: (declare such variables globally with a default definition)
    //
    {
        init_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
        if ((i = find_arg((char *)"-INIT_FILE", argc, argv)) > 0) strcpy(init_file, argv[i + 1]);
        else strcpy(init_file, (char *)"Initialization/init.bin");

        forced_dims_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
        if ((i = find_arg((char *)"-DIMS_FILE", argc, argv)) > 0) strcpy(forced_dims_file, argv[i + 1]);
        else strcpy(forced_dims_file, (char *)"Params/forced_up_to_300");

        polarities_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
        if ((i = find_arg((char *)"-POLS_FILE", argc, argv)) > 0) strcpy(polarities_file, argv[i + 1]);
        else strcpy(polarities_file, (char *)"Params/positive_all");

        forced_word_ids_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
        if ((i = find_arg((char *)"-FORCEDIDS_FILE", argc, argv)) > 0) strcpy(forced_word_ids_file, argv[i + 1]);
        else strcpy(forced_word_ids_file, (char *)"Params/forced_words_roget_300");

        k_vals_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
        if ((i = find_arg((char *)"-KVALS_FILE", argc, argv)) > 0) strcpy(k_vals_file, argv[i + 1]);
        else strcpy(k_vals_file, (char *)"Params/k_0.1_all");
    }

    vocab_size = 0;
    fid = fopen(vocab_file, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
    fclose(fid);
    
    if(forcing_enabled) return get_forced_dims();
    else{
        numForcedDims = 0;
        return train_glove();
    }
}
