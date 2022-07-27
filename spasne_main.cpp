/* Please see the "LICENSE" file for the copyright information. */

/*
Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
*/

/* Please contact Chen.Tang@UTSouthwestern.edu for programming questions about 
   this file. */

#include <iostream>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "spasne.h"

using namespace std;

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, max_iter;
	int N0, D0; 
	double alpha, beta, perplexity, theta, exaggeration;
	double *data = NULL, *pixels = NULL;
	int rand_seed = -1;

    // Read the parameters and the dataset
	if(TSNE::load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter, &pixels, &N0, &D0, &alpha, &beta, &exaggeration))
	{
		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = (double*) malloc(N * no_dims * sizeof(double));
		double* costs = (double*) calloc(N, sizeof(double));
        if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 250, 250, pixels, N0, D0, alpha, beta, exaggeration);

		// Save the results
		TSNE::save_data(Y, landmarks, costs, N, no_dims);
		
		printf("origN = %d, N = %d \n", origN, N); 

        // Clean up the memory
		free(data); data = NULL;
		free(pixels); pixels = NULL; 
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
    }
}
