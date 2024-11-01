# 1. Introduction

As an extension of the t-distributed stochastic neighbor embedding (t-SNE), the 
spatially resolved t-SNE (SpaSNE) was designed to preserve both the global gene 
expression and the spatial structure for the spatially resolved profiling data. 
By leveraging both the gene expression and the spatial information, SpaSNE gives
a comprehensive low-dimensional visualization that could best reflect the 
molecular similarities of cells and the spatial interactions between cells.  

Paper: Dimensionality reduction for visualizing spatially resolved profiling 
        data using SpaSNE.

# 2. Result

Below is an example of the SpaSNE visualization for the mouse visual Cortex 
data.

![Fig](/Image/mouse_visualCortex_annotation_and_SpaSNE_result.png)

𝑟1: Pearson correlation coefficient between pairwise Euclidean gene expression 
    distances and embedding distances of points, which was used to measure 
    gene expression preservation. 

𝑟2: Pearson correlation coefficient between pairwise spatial position distances 
    and embedding distances of points, which was used to measure spatial 
    structure preservation.
    
# 3. Environment setup and code compilation

__3.1. Download the package__

The package can be downloaded by running the following command in the terminal:
```
git clone https://github.com/Lin-Xu-lab/SpaSNE.git
```
Then, use
```
cd SpaSNE
```
to access the downloaded folder. 

If the "git clone" command does not work with your system, you can download the 
zip file from the website 
https://github.com/Lin-Xu-lab/SpaSNE.git and decompress it. Then, the folder 
that you need to access is SpaSNE-main. 

__3.2. Environment setup__

The package has been successuflly tested in a Linux environment of python 
version 3.8.8, pandas version 1.3.4, and g++ version 11.2.0. An option to set up 
the environment is to use Conda 
(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can use the following command to create an environment for SpaSNE:
```
conda create -n myenv python=3.8.8 pandas=1.3.4
```

After the environment is created, you can use the following command to activate 
it:
```
conda activate myenv
```

Please install Jupyter Notebook from https://jupyter.org/install. For example, 
you can run
```
pip install notebook
```
in the terminal to install the classic Jupyter Notebook. 

The *.cpp files need g++ to compile. The code has been successfully tested under
g++ version 11.2.0. If your g++ version cannot successfully compile the code in 
the next step, please visit https://gcc.gnu.org/ to download the GCC 11.2. 

__3.3. Compilation__

After you set up the above environment, please run the following command in the 
terminal to generate the "spasne" executable file: 
```
make
```
Please see the Makefile file for other commands. The "make" in the system that 
the software was tested is default to use GNU Make 
(https://www.gnu.org/software/make/). If using "make" cannot generate the 
"spasne" executable file in your system, please use the following command in the
terminal to generate it: 
```
g++ sptree.cpp spasne.cpp spasne_main.cpp -o spasne -O2
```  

Now you could import spasne in the current directory. 

__3.4. Import spasne in different directories (optional)__

If you would like to import spasne in different directories, there is an option 
to make it work. Please run
```
python setup.py install --user &> log
```
in the terminal and then use
```
grep Installed log
```
to obtain the path that the SpaSNE software is installed. You will see something
like 
"Installed /home/chentang/.local/lib/python3.8/site-packages/spasne-1.0-py3.8.egg".

You need to copy the "spasne" executable file generated by running "gmake" in 
the terminal to the folder that the software is installed:
```
cp spasne /home/chentang/.local/lib/python3.8/site-packages/spasne-1.0-py3.8.egg/
```
Note, in the command above, please use your installed path to replace mine. 

After doing these successfully, you are supposed to be able to import spasne 
when you are using Python or Jupyter Notebook in other folders:
```
import spasne
```

The entire flow of "Environment setup and code compilation" took less than five
minutes on my local computer. 

__3.5. Build the Docker Image (optional)__

There is an option to build a Docker image using the provided Dockerfile. You 
can use the following command in the terminal to build the Docker image:
```
docker build -t spasne .
```

Once the image is built, you can use the following command:
```
docker run -it spasne bash
```
to run the container and start using SpaSNE.

# 4. Parameter setting

SpaSNE incorporates two pivotal parameters: the global gene expression weight 𝛼, which modulates the balance between local and global gene expression preservation, and the spatial weight 𝛽, which adjusts the balance between gene expressions and spatial structure. An increase in 𝛼 results in a larger rg and a smaller rs, whereas a higher 𝛽 enhances rs and diminishes rg. Therefore, maintaining an appropriate balance between 𝛼 and 𝛽 is crucial for achieving satisfactory preservation of both gene expressions and spatial configurations. We have devised a two-stage heuristic screening approach to identify the optimal parameters for specific datasets and have included a detailed tutorial using human breast cancer data in the [Tutorials](https://github.com/Lin-Xu-lab/SpaSNE/tree/main/Tutorials) folder.

In the latest version of SpaSNE, default settings are 𝛼 = 10 and 𝛽 = 5 when spatial information input is accessible, and 𝛼 = 5 and 𝛽 = 0 when it is absent. For details on additional parameters, please refer to the [spasne.py file](https://github.com/Lin-Xu-lab/SpaSNE/tree/main/spasne.py) file.

# 5. Examples

Please refer to the [Tutorials](https://github.com/Lin-Xu-lab/SpaSNE/tree/main/Tutorials) folder for detailed examples.

# 6. Contact information

Please contact our team if you have any questions:

Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)

Chen Tang (Chen.Tang@UTSouthwestern.edu)

Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Chen Tang for programming questions about the spasne.py and *.cpp
files.

# 7. Copyright information 

The SpaSNE software uses the BSD 3-clause license. Please see the "LICENSE" file
for the copyright information. 

Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
