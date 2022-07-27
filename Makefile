# After you set up the environment, please run the following command in the 
# terminal to generate the "spasne" executable file.
# ```
# gmake
# ```
# Please run the following command in the terminal to remove the "spasne" 
# executable file.
# ```
# gmake clean
# ```
# Please run the following command in the terminal to remove both the 
# "spasne" executable file and the pycache folder.
# ```
# gmake clean0
# ```

# All the above commands are for operations in the current folder. 

spasne: spasne_main.cpp spasne.cpp sptree.cpp
	g++ sptree.cpp spasne.cpp spasne_main.cpp -o spasne -O2

clean:
	-rm spasne
	
clean0:
	-rm -rf spasne __pycache__
