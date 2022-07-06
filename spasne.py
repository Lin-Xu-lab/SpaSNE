#!/usr/bin/env python

'''
This open-source software is for implementing the SpaSNE algorithm. 

Examples: 
spasne = spasne.run_spasne(data, pixels = pixels, alpha = alpha, beta = beta)

Parameter List:

Required parameters:
data: The cell-gene matrix

Important default parameters:
# Although these parameters are default, it is recommended to set them. 
pixels: The cell-pixel matrix. 
alpha: The weight parameter for the global term. 
       By default, alpha = 8.
beta: The weight parameter for the spatial term. 
      By default, alpha = 2 when there is the "pixels" input. 

Other default parameters:
exaggeration = 4
no_dims=2 # number of dimensions
perplexity=50
theta=0.5
randseed=-1 # When randseed is -1, it means randseed is not set.
verbose = False 
initial_dims = 0 # initial dimensions for PCA. 
                   By default, initial_dims = min(data.shape[1], 200).
use_pca = True # Use PCA (True) or not (False). 
                 By default, the use_pca is set to True.
max_iter =1000 # maximum number of iterations

Publication: Dimensionality reduction for visualizing spatially resolved 
             profiling data using SpaSNE.

Please contact us if you have any questions:
Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)
Chen Tang (Chen.Tang@UTSouthwestern.edu)
Xue Xiao (Xiao.Xue@UTSouthwestern.edu)
Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact C. Tang for programming quesions about *.cpp and spasne.py files.

Version: 06/28/2022

Please see the "LICENSE" file for the copyright information. 

Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
'''

import pandas as pd
from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np
import os, sys
import io

### Constants
IS_WINDOWS = True if system() == 'Windows' else False
SPASNE_BIN_PATH = path_join(dirname(__file__), 'windows', 'spasne.exe') if IS_WINDOWS else path_join(dirname(__file__), 'spasne')
assert isfile(SPASNE_BIN_PATH), ('The "{}.py" is unable to find the "spasne" '
	'binary in its directory. Please run "gmake" in that directory to generate '
	'the "spasne" binary.').format(SPASNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2014)
# https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf (Experimental Setup, page 13)
DEFAULT_NO_DIMS = 2
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 30
DEFAULT_THETA = 0.5
EMPTY_SEED = -1
DEFAULT_USE_PCA = True
DEFAULT_MAX_ITERATIONS = 1000

###

def _argparse():
    argparse = ArgumentParser('spasne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int,
                          default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float,
            default=DEFAULT_PERPLEXITY)
    # 0.0 for theta is equivalent to vanilla t-SNE
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)
    argparse.add_argument('-r', '--randseed', type=int, default=EMPTY_SEED)
    argparse.add_argument('-n', '--initial_dims', type=int, default=INITIAL_DIMENSIONS)
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'),
            default=stdout)
    argparse.add_argument('--use_pca', action='store_true')
    argparse.add_argument('--no_pca', dest='use_pca', action='store_false')
    argparse.set_defaults(use_pca=DEFAULT_USE_PCA)
    argparse.add_argument('-m', '--max_iter', type=int, default=DEFAULT_MAX_ITERATIONS)
    return argparse


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def _is_filelike_object(f):
    try:
        return isinstance(f, (file, io.IOBase))
    except NameError:
        # 'file' is not a class in python3
        return isinstance(f, io.IOBase)


def init_spasne(samples, pixels, alpha, beta, exaggeration, workdir, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS, perplexity=DEFAULT_PERPLEXITY,
            theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False, use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS):

    if use_pca:
        samples = samples - np.mean(samples, axis=0)
        cov_x = np.dot(np.transpose(samples), samples)
        [eig_val, eig_vec] = np.linalg.eig(cov_x)

        # sorting the eigen-values in the descending order
        eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

        if initial_dims > len(eig_vec):
            initial_dims = len(eig_vec)

        # truncating the eigen-vectors matrix to keep the most important vectors
        eig_vec = np.real(eig_vec[:, :initial_dims])
        samples = np.dot(samples, eig_vec)
    else:
        samples = np.dot(samples, 1.0)	

    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)
    
    pixels = np.dot(pixels, 1.0)
    
    pixels_dim = len(pixels[0])
    pixels_count = len(pixels)

    # Note: The binary format used by spasne is roughly the same as for
    #   vanilla tsne
    with open(path_join(workdir, 'data.dat'), 'wb') as data_file:
        # Write the spasne header
        data_file.write(pack('iiddii', sample_count, sample_dim, theta, perplexity, no_dims, max_iter))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        
        data_file.write(pack('iii', randseed, pixels_count, pixels_dim))
        
        for pixel in pixels:
            data_file.write(pack('{}d'.format(len(pixel)), *pixel))
            
        data_file.write(pack('ddd', alpha, beta, exaggeration))

def load_data(input_file):
    # Read the data, using numpy's good judgement
    return np.loadtxt(input_file)

def spasne(workdir, verbose=False):

    # Call spasne and let it do its thing
    with open(devnull, 'w') as dev_null:
        spasne_p = Popen((abspath(SPASNE_BIN_PATH), ), cwd=workdir,
                # spasne is very noisy on stdout, tell it to use stderr
                #   if it is to print any output
                stdout=stderr if verbose else dev_null)
        spasne_p.wait()
        assert not spasne_p.returncode, ('ERROR: Call to spasne exited '
                'with a non-zero return code exit status, please ' +
                ('enable verbose mode and ' if not verbose else '') +
                'refer to the spasne output for further details')

    # Read and pass on the results
    with open(path_join(workdir, 'result.dat'), 'rb') as output_file:
        # The first two integers are just the number of samples and the
        #   dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file)
            for _ in range(result_samples)]
        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]
        # Put the results in order and yield it
        results.sort()
        for _, result in results:
            yield result
        # The last piece of data is the cost for each sample, we ignore it
        #read_unpack('{}d'.format(sample_count), output_file)
        
def run_spasne(data, pixels = None, alpha = 8.0, beta = 2.0, exaggeration = 4, no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=False, initial_dims=0, use_pca=True, max_iter=1000):

    if initial_dims == 0:
    	initial_dims = min(data.shape[1], 200)
    	
    if pixels is None:
        pixels_count = data.shape[0]
        pixels = pd.DataFrame(np.zeros((pixels_count,0)))
        pixels.index = data.index
        pixels['x_pixel'] = 10000.0
        pixels['y_pixel'] = 10000.0
        beta = 0.0

    # spasne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    tmp_dir_path = mkdtemp()
#   print(tmp_dir_path)

    # Load data in forked process to free memory for actual spasne calculation
    child_pid = os.fork()
    if child_pid == 0:
        if _is_filelike_object(data):
            data = load_data(data)

        init_spasne(data, pixels, alpha, beta, exaggeration, tmp_dir_path, no_dims=no_dims, perplexity=perplexity, theta=theta, randseed=randseed,verbose=verbose, initial_dims=initial_dims, use_pca=use_pca, max_iter=max_iter)
        os._exit(0)
    else:
        try:
            os.waitpid(child_pid, 0)
        except KeyboardInterrupt:
            print("Please run this program directly from python and not from ipython or jupyter.")
            print("This is an issue due to asynchronous error handling.")

        res = []
        for result in spasne(tmp_dir_path, verbose):
            sample_res = []
            for r in result:
                sample_res.append(r)
            res.append(sample_res)
        rmtree(tmp_dir_path)
        
        return np.asarray(res, dtype='float64')

def main(args):
    parser = _argparse()

    if len(args) <= 1:
        print(parser.print_help())
        return 

    argp = parser.parse_args(args[1:])
    
    for result in run_spasne(argp.input, no_dims=argp.no_dims, perplexity=argp.perplexity, theta=argp.theta, randseed=argp.randseed,
            verbose=argp.verbose, initial_dims=argp.initial_dims, use_pca=argp.use_pca, max_iter=argp.max_iter):
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'
        argp.output.write(fmt.format(*result))

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
