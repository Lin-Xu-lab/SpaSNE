/* Please see the "LICENSE" file for the copyright information. */

/*
Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
*/

#ifndef TSNE_H
#define TSNE_H

#ifdef __cplusplus
extern "C" {
namespace TSNE {
#endif
    void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
             bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
             double* pixel, int N0, int D0, double alpha, double beta, double exaggeration);
    bool load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter, double** pixels, int* n0, int* d0, double* alpha, double* beta, double* exaggeration);
    void save_data(double* data, int* landmarks, double* costs, int n, int d);
#ifdef __cplusplus
};
}
#endif

#endif
