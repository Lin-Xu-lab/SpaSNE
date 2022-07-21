# SpaSNE

As an extension of the t-distributed stochastic neighbor embedding (t-SNE), the 
spatially resolved t-SNE (SpaSNE) was designed to preserve both the global gene 
expression and the spatial structure for the spatially resolved profiling data. 
By leveraging both the gene expression and the spatial information, SpaSNE gives
a comprehensive low-dimensional visualization that could best reflect the 
molecular similarities of cells and the spatial interactions between cells.  

Paper: Dimensionality reduction for visualizing spatially resolved profiling 
        data using SpaSNE.

# Result

Below is an example of the SpaSNE visualization for the mouse visual Cortex 
data.

![Fig](/images/mouse_visualCortex_annotation_and_SpaSNE_result.png)

𝑟1: Pearson correlation coefficient between pairwise Euclidean gene expression 
    distances and embedding distances of points, which was used to measure 
    gene expression preservation. 

𝑟2: Pearson correlation coefficient between pairwise spatial position distances 
    and embedding distances of points, which was used to measure spatial 
    structure preservation;
    
# Environment setup and code compilation

The code has been successuflly tested in an environment of python version 
3.8.8 and pandas version 1.3.4. It may also work for the environment of 
your machine. However, if you find the code get stuck there, we recommend you
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

SpaSNE has two key parameters: The global gene expression weight 𝛼 that balances 
the prevervations of local and global gene expressions, and the spatial weight 𝛽 
that balances gene expressions and the spatial structure. A larger 𝛼 leads to a 
larger 𝑟1 and a smaller 𝑟2, while a larger 𝛽 leads to a larger 𝑟2 and a smaller 
𝑟1. Thus, a proper ratio between 𝛼 and 𝛽 is required to give a satisfied 
preservation of both the gene expressions and the spatial structure. We give the 
following recommendations for setting 𝛼 and 𝛽:  

𝛼 ∈ [5,15], 𝛽 ∈ [1,7.5], and 𝛼 / 𝛽 ≥ 2. 

When the random seed is variable or not set, a large 𝛼 may result in a large 
chance of embedding failure, especially for small (spatial samples) datasets. 
For example, for the BreastCancer1272 dataset used as an example in this 
software package, there was a higher chance of embedding failure when setting 𝛼 
to 12 than setting it to 10 (𝛽 was also changed accordingly based on an 𝛼 / 𝛽 
ratio). Therefore, for small datasets, it might be relatively stable when 
setting 𝛼 not larger than 10. 

In this software package, the default parameters have been set as (𝛼 = 8, 𝛽 = 2)
when the input of the spatial information is available, and (𝛼 = 8, 𝛽 = 0) when 
there is no input of the sptial information. 

Please see the spasne.py file for the introduction of more parameters. 

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
		example. Before running this file, please type the following command in
		the terminal to obtain the "data" folder: 
```
tar xvzf data.tar.gz
```
		
Below, we show an example for the mouse visual Cortex data. First, please type
```
cd spasne-examples
```
in the terminal to enter the "spasne-examples" folder.

Then, type
```
jupyter notebook &
```
to open the jupyter notebook. Left click the 
spasne_VisualCortex1207_example.ipynb file to open it. 

Run the code below:
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
Visualizing spots from image
```
matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(1, 1, figsize=(6,5))
sz = 100

plot_color=['#911eb4', '#46f0f0','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',  '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']
domains="gt"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'Mouse visual cortex'
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')
ax.axes.invert_yaxis()
```
![Fig](/images/mouse_visualCortex_annotation.png)

Calculating data distances and spatial distances
```
N = df_data.shape[0]
X = np.array(df_data)
dist_sq = euclidean_distances(X, X)
dist_sq = (dist_sq + dist_sq.T) / 2.0
dist_data = scipy.spatial.distance.squareform(dist_sq)
X_spa = np.array(df_pixel)
dist_sq = euclidean_distances(X_spa,X_spa)
dist_sq = (dist_sq + dist_sq.T) / 2.0
dist_spatial = scipy.spatial.distance.squareform(dist_sq)
df_pixel = df_pixel.astype(np.float64)
```
Performing t-SNE embedding
```
tsne_pos = spasne.run_spasne(df_data, alpha = 0.0, randseed = 5)
dist_sq = euclidean_distances(tsne_pos, tsne_pos)
dist_sq = (dist_sq + dist_sq.T)/2
dist_model = scipy.spatial.distance.squareform(dist_sq)
# Measuring gene expression presrvation
(r1,_) = scipy.stats.pearsonr(dist_data, dist_model)
# Measuring spatial structure presrvation
(r2,_) = scipy.stats.pearsonr(dist_spatial, dist_model)
# Calculating silhouette score based on ground truth annotations
ss = sklearn.metrics.silhouette_score(tsne_pos,cluster_label)
quant_eval_tsne = [r1,r2,ss]
adata.obs['tsne_pos_x'] = tsne_pos[:,0]
adata.obs['tsne_pos_y'] = tsne_pos[:,1]

```
Visualizing spots from t-SNE embedding
```
matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(1, 1, figsize=(6,5))
domains="gt"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 't-SNE, ' + 'r1 = %.2f'% quant_eval_tsne[0] + ', r2 = %.2f'%quant_eval_tsne[1] + ', s = %.2f'%quant_eval_tsne[2]
ax=sc.pl.scatter(adata,alpha=1,x="tsne_pos_x",y="tsne_pos_y",color=domains,title=titles,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')

```
(-7.451521853408612, 9.16820293951664, -11.582022822788456, 9.920880900761276)
![Fig](/images/mouse_visualCortex_t-SNE_result.png)

Performing SpaSNE embedding
```
alpha = 9.0
beta = 2.25
spasne_pos = spasne.run_spasne(df_data, pixels = df_pixel, alpha = alpha, beta = beta, randseed = 5)
dist_sq = euclidean_distances(spasne_pos, spasne_pos)
dist_sq = (dist_sq + dist_sq.T)/2
dist_model = scipy.spatial.distance.squareform(dist_sq)
(r1,_) = scipy.stats.pearsonr(dist_data, dist_model)
(r2,_) = scipy.stats.pearsonr(dist_spatial, dist_model)
ss = sklearn.metrics.silhouette_score(spasne_pos,cluster_label)
quant_eval_spasne = [r1,r2,ss]
adata.obs['spasne_pos_x'] = spasne_pos[:,0]
adata.obs['spasne_pos_y'] = spasne_pos[:,1]
```
Visualizing spots from SpaSNE embedding
```
matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(1, 1, figsize=(6,5))
domains="gt"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'spaSNE, ' + 'r1 = %.2f'% quant_eval_spasne[0] + ', r2 = %.2f'%quant_eval_spasne[1] + ', s = %.2f'%quant_eval_spasne[2]
ax=sc.pl.scatter(adata,alpha=1,x="spasne_pos_x",y="spasne_pos_y",color=domains,title=titles,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')
ax.axes.invert_xaxis()

```
![Fig](/images/mouse_visualCortex_spaSNE_result.png)

# Contact information

Please contact our team if you have questions:

Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)

Chen Tang (Chen.Tang@UTSouthwestern.edu)

Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Chen Tang for programming questions about the spasne.py and *.cpp
files.

# Copyright information 

Please see the "LICENSE" file for the copyright information. 

Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
