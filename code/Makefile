CC=nvcc
CFLAGS=-std=c++17 -arch sm_35 -O3

all: main.cu sparsemat_cuda.cpp
	$(CC) $(CFLAGS) main.cu sparsemat_cuda.cpp -o a4

clean:
	rm a4
