# SpaSNE

As an extension of the t-distributed stochastic neighbor embedding (t-SNE), the 
spatially resolved t-SNE (SpaSNE) was designed to preserve both the global gene 
expression and the spatial structure for the spatially resolved profiling data. 
By leveraging both the gene expression and the spatial information, SpaSNE 
presents a comprehensive visualization in the low-dimensional space, which best 
reflects the molecular similarities of cells and the spatial interactions 
between cells. 

# Result

Below is an example of the SpaSNE visualization for the mouse visual Cortex 
data.

![mouse_visualCortex_image_and_SpaSNE_result](/images/mouse_visualCortex_image_and_SpaSNE_result.png)

𝑟1: Pearson correlation coefficient between the gene expression's pairwise 
    Euclidean distances and the SpaSNE points' distances in the embedding 
    space.

𝑟2: Pearson correlation coefficient between the spatial locations' pairwise 
    Euclidean distances and the SpaSNE points' distances in the embedding 
    space. 
    
# Running environment and code compilation

The code has been successuflly tested in an environment of python version 
3.8.8 and pandas version 1.3.4. It may also work for the environment of 
your machine. However, if you find the code stuck there, we recommend you
to set up the environment as the same with our successful test. An option 
to set up the environment is to use Conda 
(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can use the following command to create an enviroment for SpaSNE:
```
conda create -n myenv python=3.8.8 pandas=1.3.4 jupyter
```
You can use the folloing command to activate it:
```
conda activate myenv
```

After you set up the environment, please run the following command in the 
terminal to generate the "spasne" executable file.
```
gmake
```

Please run the following command in the terminal to remove the "spasne" 
executable file.
```
gmake clean
```

Please run the following command in the terminal to remove both the 
"spasne" executable file and the "__pycache__" folder.
```
gmake clean0
```

# Parameter setting

SpaSNE has two key parameters: the global gene expression weight 𝛼 
which balances local and global gene expression preservation, and the 
spatial weight 𝛽 which balances gene expression and spatial structure 
preservation. Larger 𝛼 leads to larger 𝑟1 and smaller 𝑟2, while larger 
𝛽 leads to larger 𝑟2 and smaller 𝑟1. Thus, a proper ratio between 𝛼 and 𝛽 
is required to give a good preservation of both gene expression 
preservation and spatial structure preservation. In addition to the 
relative ratio between 𝛼 and 𝛽, we noticed that the magnitude of 𝛼 
influences the reproducibility of the embedding. Larger 𝛼 results in 
higher chance of failure of embedding, especially when the data size is 
small. Based on the above considerations and the experiences on five 
real datasets, we gave the following recommendations for setting 𝛼 and 𝛽: 

𝛼 ∈ [6,10],  𝛽 ∈ [1,5],  𝛼/𝛽 ≥ 2. 

# Examples

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
		
Below, we show an example for the mouse visual Cortex data. First, please type
```
spasne-examples
```
in the terminal to enter the "spasne-examples" folder.

Then, type
```
jupyter notebook &
```
to open the jupyter notebook. Click the spasne_VisualCortex1207_example.ipynb file to
open it. 

```
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import scipy
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
sys.path.append("..")
import spasne
```
Loading data and transforming it to AnnData object
```
df_data = pd.read_csv('data/mouse_VisualCortex1207_data_pc200.csv',sep=",",header=0,na_filter=False,index_col=None) 
df_pixel = pd.read_csv('data/mouse_VisualCortex1207_pixels.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_labels = pd.read_csv('data/mouse_VisualCortex1207_labels.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_PCs = pd.DataFrame(list(df_data.columns), index = df_data.columns, columns =['PCs'] )
cluster_label = list(df_labels['LayerName'])
adata = sc.AnnData(X = df_data, obs = df_pixel, var = df_PCs)
adata.obs['gt'] = cluster_label
```
/home/chentang/anaconda3/lib/python3.8/site-packages/anndata/_core/anndata.py:120: ImplicitModificationWarning: Transforming to str index.
  warnings.warn("Transforming to str index.", ImplicitModificationWarning)

Visualizing spots from image
	
# Copyright information

Please see the "LICENSE" file for the copyright information. 

Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
