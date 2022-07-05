spasne: spasne_main.cpp spasne.cpp sptree.cpp
	g++ sptree.cpp spasne.cpp spasne_main.cpp -o spasne -O2

clean:
	-rm -f spasne
	
clean0:
	-rm -rf spasne __pycache__
