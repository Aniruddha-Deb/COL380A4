#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void test(){
    printf("Hi Cuda World\n");
}

int main( int argc, char** argv )
{
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
