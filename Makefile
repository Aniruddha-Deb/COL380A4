CC=nvcc
CFLAGS=-std=c++17 -arch sm_35 -O3

all: code/main.cu code/sparsemat_cuda.cpp
	$(CC) $(CFLAGS) code/main.cu code/sparsemat_cuda.cpp -o a4

clean:
	rm a4
