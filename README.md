# SpaSNE

python_version = 3.8.8

pandas_version = 1.3.4

Please run "gmake" in the terminal to generate the "spasne" executable file.

Please run "gmake clean" in the terminal to remove the "spasne" executable file.

Please run "gmake clean0" in the terminal to remove both the "spasne" executable
file and the "__pycache__" folder.

1. There are two SpaSNE examples in the "spasne-examples" folder:

	1.1 Please use jupyter notebook to open the 
		spasne_VisualCortex1207_example.ipynb for the Mouse Visual Cortex 
		example.

	1.2 Please use jupyter notebook to open the 
		spasne_BreastCancer1272_example.ipynb for the Human Breast Cancer 
		example.
    
2. There is one preprocessing example in the "preprocessing-example" folder:

	2.1 Please use Jupyter notebook to open the 
		preprocessing_BreastCancer1272_example.ipynb for the Human Breast Cancer 
		example. Before running this file, please type "tar xvzf data.tar.gz" in
		the terminal to obtain the "data" folder. 

Please see the "LICENSE" file for the copyright information. 

Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
