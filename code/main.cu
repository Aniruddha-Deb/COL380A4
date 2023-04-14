#include "sparsemat.hpp"
#include <cuda_runtime.h>

#define idx(i,j) (((idx_t)(i))<<32 | (j));
#define idx_i(idx) (int((idx)>>32));
#define idx_j(idx) (int(idx));

// we still don't know if we have to do A*B where A,B are aribitrary-sized or
// if we need to do A*A where A is nxn

SparseMatrix::SparseMatrix(int _n, int _m, int _k) {
    n = _n;
    m = _m;
    k = _k;
    I = new int[k];
    J = new int[k];
    M = new uint32_t*[k];
    for (int i=0; i<k; i++) {
        M[i] = new uint32_t[m*m]();
    }
}

SparseMatrix::SparseMatrix(char* filename) {

    std::ifstream input(filename, std::ios_base::binary);
    input.read((char*)&n, 4);
    input.read((char*)&m, 4);
    input.read((char*)&k, 4);

    I = new int[k];
    J = new int[k];
    M = new uint32_t*[k];
    for (int i=0; i<k; i++) {
        M[i] = new uint32_t[m*m]();
    }

    uint32_t t;
    for (int ii=0; ii<k; ii++) {
        input.read((char*)&I[ii], 4);
        input.read((char*)&J[ii], 4);
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                input.read((char*)&t, 2);
                M[ii][m*r+c] = t;
            }
        }
    }
}


void SparseMatrix::print() {
    // convert to a dense matrix and print
    int *mat = new int[n*n]();
    for (int b=0; b<k; b++) {
        for (int r=0; r<m; r++) {
            memcpy(&mat[n*(I[b]*m+r) + (J[b]*m)], &M[b][m*r], sizeof(int)*m);
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << mat[n*i+j] << " ";
        }
        std::cout << std::endl;
    }
}


void SparseMatrix::save(char* filename, int nbytes) {

    std::ofstream output(filename, std::ios_base::binary);
    output.write((char*)&n, 4);
    output.write((char*)&m, 4);
    output.write((char*)&k, 4);

    for (int ii=0; ii<k; ii++) {
        output.write((char*)&I[ii], 4);
        output.write((char*)&J[ii], 4);
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // TODO truncation, but that's ok for now
                output.write((char*)&M[ii][m*r+c], nbytes);
            }
        }
    }
}


SparseMatrix::~SparseMatrix() {
    delete I;
    delete J;
    for (int i=0; i<k; i++) {
        delete M[i];
    }
    delete M;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int m> __global__ 
void MatrixMulCUDA(uint32_t *C, uint32_t *A, uint32_t *B) {

    __shared__ uint32_t As[m][m];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

SparseMatrix* sparse_matrix_multiply(SparseMatrix *A, SparseMatrix *B) {

    
    // allocate device memory

    // run the kernel on 
    // n is upto 2^15, m is either 2^2 or 2^3 so n/m <= 2^13
    // kernel supports 2^16 block size tops, so it's easy to parallelize 
    // everything all at once. If a block is empty, then we'll just leave
    // it out.

    // setup a stream for the matrix. Initialize d_A,d_B,d_C on the device and
    // then copy over essential information first, then move the actual matrix
    // over...

    // use thrust maps to figure out which blocks are along which coordinates.
    // Store these maps on the matrix itself to make multiplication faster.
    // 
    // NOPE, thrust doesn't have maps... we need to use a CSR/CSC mat for that.
    // each block will be responsible for computing all elements of that block.

    // how do we bookkeep? we'll need to mark out and allocate space for empty 
    // blocks and add the indices to the final CSR matrix as well. Don't need
    // conflicts here. Can do this in the final synchronize loop.
    // maybe can try using a thrust vector here for C

    // DCSR may be one option
    // given the constraints, I think the easiest option is to allocate a 
    // large 2-D map and then parallelly allocate pointers on it.
    // Then reduce the map in parallel 

    CUDA
}


/**
 * Run a simple test of matrix multiplication using CUDA
 */

int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));
    cudaStream_t stream;

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<<grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32> <<<grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                               static_cast<double>(dimsA.y) *
                               static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                               dimsB.x, dimsB.y);

    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}

