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
#include "vptree.h"
#include "sptree.h"
#include "spasne.h"

using namespace std;

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

static void zeroMean(double* X, int N, int D);
static void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);

static void computeGaussianPerplexity_L2(double* X, int N, int D, double* P);

static double randn();
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
static void computeExactGradient_SpaSNE(double* P, double* P_global, double* P_space, double* Y, int N, int D, double* dC, double alpha, double beta); 
static void computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
static void computeGradient_SpaSNE(double* Y, int N, int D, double* dC, double theta, double* P, double* P_global, double* P_space, double alpha, double beta); 
static double evaluateError_exact(double* P, double* Y, int N, int D);
static double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta);
static double evaluateError_exact_SpaSNE(double* P, double* Y, int N, int D, double* P_global, double* P_space, double alpha, double beta); 
static double evaluateError_SpaSNE(double* P, double* Y, int N, int D, double theta, double* P_global, double* P_space, double alpha, double beta);
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
static void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N);

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
               bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
               double* pixels, int N0, int D0, double alpha, double beta, double exaggeration)
{
    // Set random seed
    if (skip_random_init != true)
    {
		if(rand_seed >= 0) 
		{
			printf("Using random seed: %d\n", rand_seed);
			srand((unsigned int) rand_seed);
		}
		else 
		{
			printf("Using current time as random seed...\n");
			srand(time(NULL) % 100);
		}
    }

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) 
    {
    	printf("Perplexity too large for the number of data points!\n"); 
    	exit(1);
    }
    
    printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    
    bool exact = (theta == .0) ? true : false;

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
	double momentum = 0.5, final_momentum = 0.8;
	double eta = 500.0;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    
    if(dY == NULL || uY == NULL || gains == NULL)
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    for(int i = 0; i < N * no_dims; i++)
    	uY[i] =  0.0;
    	
    for(int i = 0; i < N * no_dims; i++)
    	gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    printf("Computing input similarities...\n");
    
    start = clock();
    
    zeroMean(X, N, D);
    
    double max_X = .0;
    
    for(int i = 0; i < N * D; i++)
    {
        if(fabs(X[i]) > max_X) 
        	max_X = fabs(X[i]);
    }
    
    for(int i = 0; i < N * D; i++) 
    	X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    double* P; unsigned int* row_P; unsigned int* col_P; double* val_P;
/*    
    if(exact)
    {
        // Compute similarities
        printf("Exact?");
        P = (double*) malloc(N * N * sizeof(double));
        
        if(P == NULL) 
        {
        	printf("Memory allocation failed!\n"); 
        	exit(1);
        }
        
        computeGaussianPerplexity(X, N, D, P, perplexity);

        // Symmetrize input similarities
        printf("Symmetrizing...\n");
        int nN = 0;
        
        for(int n = 0; n < N; n++)
        {
            int mN = (n + 1) * N;
            
            for(int m = n + 1; m < N; m++) 
            {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            
            nN += N;
        }
        double sum_P = .0;
        
        for(int i = 0; i < N * N; i++) 
        	sum_P += P[i];
        	
        for(int i = 0; i < N * N; i++) 
        	P[i] /= sum_P;
    }

    // Compute input similarities for approximate t-SNE
    else
    {
        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity));

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        
        double sum_P = .0;
        
        for(int i = 0; i < row_P[N]; i++) 
        	sum_P += val_P[i];
        	
        for(int i = 0; i < row_P[N]; i++)
        	val_P[i] /= sum_P;
    }
*/
//============================================
	printf("calculate the local term \n");
	
	double sum_P = .0;

    P = (double*) malloc(N * N * sizeof(double));
    
    if(P == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1);
    }
    
    computeGaussianPerplexity(X, N, D, P, perplexity);

    // Symmetrize input similarities
    printf("Symmetrizing...\n");
    int nN = 0;
    
    for(int n = 0; n < N; n++)
    {
        int mN = (n + 1) * N;
        
        for(int m = n + 1; m < N; m++) 
        {
            P[nN + m] += P[mN + n];
            P[mN + n]  = P[nN + m];
            mN += N;
        }
        
        nN += N;
    }
    sum_P = .0;
    
    for(int i = 0; i < N * N; i++) 
    	sum_P += P[i];
    	
    for(int i = 0; i < N * N; i++) 
    	P[i] /= sum_P;

//============================================
	printf("calculate the global term \n");

	double* P_global = NULL;
	P_global = (double*) malloc(N * N * sizeof(double));

	if(P_global == NULL) 
	{
		printf("Memory allocation failed!\n"); 
		exit(1);
	}

	computeGaussianPerplexity_L2(X, N, D, P_global);
	//============================================
	printf("calculate the spatial term \n");

	double* P_space = NULL;
	P_space = (double*) malloc(N * N * sizeof(double));

	if(P_space == NULL) 
	{
		printf("Memory allocation failed!\n"); 
		exit(1);
	}

	computeGaussianPerplexity_L2(pixels, N, D0, P_space);
    
    end = clock();

    // Lie about the P-values
/*  if(exact) 
    	for(int i = 0; i < N * N; i++)
    		P[i] *= 12.0; 
    else 
    	for(int i = 0; i < row_P[N]; i++) 
    		val_P[i] *= 12.0;
*/    		

	for(int i = 0; i < N * N; i++)
	{
		P[i] *= exaggeration; 
	}

	// Initialize solution (randomly)
	if (skip_random_init != true)
	{
  		for(int i = 0; i < N * no_dims; i++)
  			Y[i] = randn() * .0001;
	}

	// Perform main training loop
/*    if(exact) 
    	printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    else 
    	printf("Input similarities computed in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N)); */
    	
    printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    	
    start = clock(); 
    
    printf("exact = %d \n", exact); 
    
    if(exact) 
    	printf("Run Exact! \n"); 
    else 
    	printf("Run Barnes-Hut algorithm! \n"); 

	for(int iter = 0; iter < max_iter; iter++) 
	{
	
	//	printf("iter = %d \n", iter); 
        // Compute (approximate) gradient
        if(exact) 
       // 	computeExactGradient(P, Y, N, no_dims, dY);
        	computeExactGradient_SpaSNE(P, P_global, P_space, Y, N, no_dims, dY, alpha, beta); 
        else 
       // 	computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta); 
       		computeGradient_SpaSNE(Y, N, no_dims, dY, theta, P, P_global, P_space, alpha, beta);
        	
 /*       if(exact) 
        	computeExactGradient_SpaSNE(P, P_global, P_space, Y, N, no_dims, dY, alpha, beta); 
        else 
        	computeGradient_SpaSNE(Y, N, no_dims, dY, theta, P, P_global, P_space, alpha, beta); */

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .15) : (gains[i] * .85);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
		zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
/*        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
            else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
        }	*/
        
/*        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
        }	*/	
        
        if(iter == stop_lying_iter)
            for(int i = 0; i < N * N; i++)
            {
            	P[i] /= exaggeration;
        	}	
        
        
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1))
        {
            end = clock();
            
            double C = .0;
            
            if(exact) 
            	C = evaluateError_exact_SpaSNE(P, Y, N, no_dims, P_global, P_space, alpha, beta);
            else
            	C = evaluateError_SpaSNE(P, Y, N, no_dims, theta, P_global, P_space, alpha, beta);
            	
            if(iter == 0)
                printf("Iteration %d: error is %f\n", iter + 1, C);
            else 
            {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);
            }
            
			start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    
    if(exact) 
    {
    	free(P); P = NULL; 
    	free(P_global); P_global = NULL; 
    	free(P_space); P_space = NULL; 
    }
    else 
    {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    
    printf("Fitting performed in %4.2f seconds.\n", total_time);
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
static void computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
/*
static void computeGradient_SpaSNE(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, double * P_global, double *P_space, double alpha, double beta)
{
    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = 0.0, sum_Q_global = 0.0; 
    double* pos_f = NULL, pos_f2 = NULL, neg_f = NULL, neg_f2 = NULL; 
    pos_f = (double*) calloc(N * D, sizeof(double));
    neg_f = (double*) calloc(N * D, sizeof(double));
    pos_f2 = (double*) calloc(N * D, sizeof(double));
    neg_f2 = (double*) calloc(N * D, sizeof(double));
    
    if(pos_f == NULL || neg_f == NULL || pos_f2 == NULL || neg_f2 == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    tree->computeEdgeForces_SpaSNE(inp_row_P, inp_col_P, inp_val_P, N, pos_f, P_global, P_Space, alpha, beta);
    
    for(int n = 0; n < N; n++) 
    	tree->computeNonEdgeForces_SpaSNE(n, theta, neg_f + n * D, &sum_Q, &sum_Q_global);

    // Compute final SpaSNE gradient
    for(int i = 0; i < N * D; i++)
    {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q - (alpha + beta) * neg_f_global[i] / sum_Q_global);
    }
    
    free(pos_f);
    free(neg_f);
    delete tree;
}
*/

static void computeGradient_SpaSNE(double* Y, int N, int D, double* dC, double theta, double* P, double* P_global, double* P_space, double alpha, double beta)
{
    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = 0.0, sum_Q_global = 0.0; 
    double *pos_f = NULL, *neg_f = NULL, *neg_f_global = NULL; 
    pos_f = (double*) calloc(N * D, sizeof(double));
    neg_f = (double*) calloc(N * D, sizeof(double));
    neg_f_global = (double*) calloc(N * D, sizeof(double));
    
    if(pos_f == NULL || neg_f == NULL || neg_f_global == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
//    tree->computeEdgeForces_SpaSNE(inp_row_P, inp_col_P, inp_val_P, N, pos_f, P_global, P_Space, alpha, beta);
	tree->computeEdgeForces_SpaSNE(N, pos_f, P, P_global, P_space, alpha, beta);
    
    for(int n = 0; n < N; n++) 
    	tree->computeNonEdgeForces_SpaSNE(n, theta, neg_f + n * D, &sum_Q, neg_f_global + n * D, &sum_Q_global);

    // Compute final SpaSNE gradient
    for(int i = 0; i < N * D; i++)
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q - (alpha + beta) * neg_f_global[i] / sum_Q_global);
    
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                for(int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
		}
        nN += N;
        nD += D;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}

// Compute gradient of the t-SNE cost function (exact)
static void computeExactGradient_SpaSNE(double* P, double* P_global, double* P_space, double* Y, int N, int D, double* dC, double alpha, double beta) 
{
	// Make sure the current gradient contains zeros
	memset(dC, 0, N * D * sizeof(double)); 

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1);
    }
    
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    double* Q_global = (double*) malloc(N * N * sizeof(double));
    if(Q_global == NULL) 
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    double sum_Q = 0.0, sum_Q_global = 0.0;
    int nN = 0;
    
    for(int n = 0; n < N; n++)
    {
    	for(int m = 0; m < N; m++)
    	{
            if(n != m)
            {
                Q[nN + m] = 1.0 / (1.0 + DD[nN + m]);
                sum_Q += Q[nN + m];
                
                Q_global[nN + m] = 1.0 + DD[nN + m]; 
                sum_Q_global += Q_global[nN + m]; 
            }
            else
            {
            	Q[nN + m] = 0.0; 
            	Q_global[nN + m] = 0.0; 
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) 
	{
        int mD = 0;
    	for(int m = 0; m < N; m++) 
    	{
            if(n != m) 
            {
          	 	double mult = ((P[nN + m] - (Q[nN + m] / sum_Q)) 
          	 				- alpha * (P_global[nN + m] - Q_global[nN + m] / sum_Q_global)
          	 				- beta * (P_space[nN + m] - Q_global[nN + m] / sum_Q_global)) * Q[nN + m]; 
                				
                for(int d = 0; d < D; d++) 
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
            }
            
            mD += D;
		}
		
        nN += N;
        nD += D;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
    free(Q_global); Q_global = NULL; 
}

// Evaluate t-SNE cost function (exactly)
static double evaluateError_exact(double* P, double* Y, int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
	for(int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

static double evaluateError_exact_SpaSNE(double* P, double* Y, int N, int D, double* P_global, double* P_space, double alpha, double beta) 
{
    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    double* Q_global = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    double sum_Q_global = DBL_MIN; 
    for(int n = 0; n < N; n++) 
    {
    	for(int m = 0; m < N; m++) 
    	{
            if(n != m) 
            {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                Q_global[nN + m] = 1 + DD[nN + m]; 
                sum_Q += Q[nN + m];
                sum_Q_global += Q_global[nN + m]; 
            }
            else 
            {
            	Q[nN + m] = DBL_MIN;
            	Q_global[nN + m] = DBL_MIN; 
            }
        }
        
        nN += N;
    }
    
    for(int i = 0; i < N * N; i++)
    {
    	Q[i] /= sum_Q;
    	Q_global[i] /= sum_Q_global; 
    }

    // Sum SpaSNE error
    double C = 0.0;
	for(int n = 0; n < N * N; n++) 
	{
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
        C += alpha * P_global[n] * log((P_global[n] + FLT_MIN) / (Q_global[n] + FLT_MIN));
        C += beta * P_space[n] * log((P_space[n] + FLT_MIN) / (Q_global[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
    free(Q_global); 
    
	return C;
}

// Evaluate t-SNE cost function (approximately)
static double evaluateError_SpaSNE(double* P, double* Y, int N, int D, double theta, double* P_global, double* P_space, double alpha, double beta)
{

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double* buff_global = (double*) calloc(D, sizeof(double));
    double sum_Q = 0.0, sum_Q_global = 0.0;
    
    for(int n = 0; n < N; n++) 
    	tree->computeNonEdgeForces_SpaSNE(n, theta, buff, &sum_Q, buff_global, &sum_Q_global);
    	
    double Q, Q_global, C = 0.0;
    unsigned int nD = 0; 
    unsigned int nN = 0; 
    
    for(unsigned int n = 0; n < N; n++) 
    {
    	unsigned int mD = 0; 
    
        for(unsigned int m = 0; m < N; m++)
	    {
	        // Compute pairwise distance and Q-value
	        Q = 0.0;
	        
	        for(unsigned int d = 0; d < D; d++) 
	        	buff[d] = Y[nD + d] - Y[mD + d];
	        	
	        for(unsigned int d = 0; d < D; d++) 
	        	Q += buff[d] * buff[d];
	        	
	        Q = (1.0 / (1.0 + Q)) / sum_Q;
	        C += P[nN + m] * log((P[nN + m] + FLT_MIN) / (Q + FLT_MIN));
	        
	        Q_global = (1.0 + Q) / sum_Q_global;
	        C += alpha * P_global[nN + m] * log((P_global[nN + m] + FLT_MIN) / (Q_global + FLT_MIN));
	        C += beta * P_space[nN + m] * log((P_space[nN + m] + FLT_MIN) / (Q_global + FLT_MIN));
	        
	        mD += D;
	    }
		    
        nN += N;
        nD += D;
    }

    // Clean up memory
    free(buff);
    free(buff_global); 
    delete tree;
    return C;
}

// Evaluate t-SNE cost function (approximately)
static double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Compute input similarities with a fixed perplexity
static void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	
    if(DD == NULL) 
    {
    	printf("Memory allocation failed!\n");
    	exit(1);
    }
    
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX){
						if (beta < 0) {
              						beta *= 2;
						} else {
						      	beta = beta <= 1.0 ? -0.5 : beta / 2.0;
						} 
					} else {
						beta = (beta + min_beta) / 2.0;
					}
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
        nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}

static void computeGaussianPerplexity_L2(double* X, int N, int D, double* P)
{
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	
    if(DD == NULL) 
    {
    	printf("Memory allocation failed!\n");
    	exit(1);
    }
    
	computeSquaredEuclideanDistance(X, N, D, DD);
	
	// Compute P-global-matrix and normalization sum
	
	double sum_P = 0.0;
	int nN = 0; 

    for(int n = 0; n < N; n++) 
    {
    	for(int m = 0; m < N; m++) 
    	{
    		P[nN + m] = 1 + DD[nN + m];
    	
            if(n != m) 
                sum_P += P[nN + m];
        }
        
        nN += N;
    }
    
    for(int i = 0; i < N * N; i++)
    	P[i] /= sum_P;
    	
	// Clean up memory
	free(DD); DD = NULL;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}


// Symmetrizes a sparse matrix
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
static void zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
static double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter, double** pixels, int* n0, int* d0, double *alpha, double *beta, double *exaggeration) 
{
	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	
	int err_fread; 
	
	err_fread = fread(n, sizeof(int), 1, h);						// number of datapoints
	err_fread = fread(d, sizeof(int), 1, h);						// original dimensionality
    err_fread = fread(theta, sizeof(double), 1, h);					// gradient accuracy
	err_fread = fread(perplexity, sizeof(double), 1, h);			// perplexity
	err_fread = fread(no_dims, sizeof(int), 1, h);          		// output dimensionality
    err_fread = fread(max_iter, sizeof(int),1, h);           		// maximum number of iterations
    
    printf("n = %d, d = %d \n", *n, *d); 

	*data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL)
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    err_fread = fread(*data, sizeof(double), (*n) * (*d), h);		// the data
    
	err_fread = fread(rand_seed, sizeof(int), 1, h);			// random seed
    	
    // Added for reading in spatial locations
    	
	err_fread = fread(n0, sizeof(int), 1, h);						// number of datapoints
	err_fread = fread(d0, sizeof(int), 1, h);						// original dimensionality 
	
	printf("n0 = %d, d0 = %d \n", *n0, *d0); 

	*pixels = (double*) malloc(*d0 * *n0 * sizeof(double));
    if(pixels == NULL)
    {
    	printf("Memory allocation failed!\n"); 
    	exit(1); 
    }
    
    err_fread = fread(*pixels, sizeof(double), (*n0) * (*d0), h);		// the pixels
    
/*    
    int index = 0;
    
    for (int i = 0; i < (*n0); i++)
    	for (int j = 0; j < (*d0); j++)
    	{
    		printf("pixel[%d][%d] = %f", i, j, *(*pixels + index++)); 
    		
    		if (j % 2 == 0)
    			printf(", ");
    		else
    			printf("\n");
    	}
*/	
    err_fread = fread(alpha, sizeof(double), 1, h);					// alpha
	err_fread = fread(beta, sizeof(double), 1, h);					// beta
	err_fread = fread(exaggeration, sizeof(double), 1, h);			// exaggeration
	
	printf("rand_seed = %d, alpha = %f, beta = %f, exaggeration = %f \n", *rand_seed, *alpha, *beta, *exaggeration); 

	fclose(h);
	
	printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen("result.dat", "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
    fwrite(costs, sizeof(double), n, h);
    fclose(h);
	printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
