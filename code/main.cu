#include "sparsemat.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

// sparse matrix multiplication C = A * B on the device 
// A is BCSR while B is BCSC
template <int m> __global__
void sparse_mul_cuda(uint32_t *A_data, int *A_idxs, int *A_idxptrs, 
        uint32_t *B_data, int *B_idxs, int *B_idxptrs, uint32_t* C_data, 
        uint8_t* C_valid, int n) {


    // this block multiplies C(bx, by) <- sum_i A(bx, i) * B(i, by)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // just return if there's nothing to do
    if (A_idxptrs[bx] == A_idxptrs[bx+1] && B_idxptrs[by] == B_idxptrs[by+1]) return;

    __shared__ uint32_t A_buf[m][m];
    __shared__ uint32_t B_buf[m][m];
    __shared__ uint32_t C_buf[m][m]();

    // use a two-pointer algorithm to find out which matrices to multiply first
    // invariants: 
    //     A_idxs[Ap] <= B_idxs[Bp]
    //     A_idxptrs[bx] <= Ap < A_idxptrs[bx+1]
    //     B_idxptrs[by] <= Bp < B_idxptrs[by+1]
    int Ap = A_idxptrs[bx];
    int Bp = B_idxptrs[by];

    uint8_t multiplied = 0;
    for (; Bp < B_idxptrs[by+1]; Bp++) {

        while (A_idxs[Ap] < B_idxs[Bp] && Ap < A_idxptrs[bx+1]) Ap++;
        if (Ap >= A_idxptrs[bx+1]) break;
        if (A_idxs[Ap] > B_idxs[Bp]) continue;

        // all invariants are satisfied here, so we can multiply
        multiplied = 1;

        // load into A_buf and B_buf 
        A_buf[tx][ty] = A_data[A_idxptrs[Ap]*m*m + tx*m + ty]
        B_buf[tx][ty] = B_data[B_idxptrs[Bp]*m*m + tx*m + ty]
        __syncthreads();

        // Do the multiplication
        #pragma unroll
        for (int i=0; i<m; i++) {
            C_buf[tx][ty] += As[tx][i] * Bs[k][ty];
            // TODO add clipping here?
        }
        __syncthreads();
    }

    if (!multiplied) return;

    // copy C_buf to C_data: each thread copies one element
    C_valid[bx*m + by] = 1;
    C_data[(bx*m + tx)*n + by*m + ty] = C_buf[tx][ty];
}

//CSRMatrix* sparse_matrix_multiply(CSRMatrix *A, CSRMatrix *B) {
uint32_t* sparse_matrix_multiply(CSRMatrix *A, CSRMatrix *B) {

    // allocate device memory

    // run the kernel on the device
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

    assert(A->ct == CT_ROW);
    assert(B->ct == CT_COL);

    // need to think of device memory allocation... Let's just assume that we 
    // get 12GB of device memory when we're running, so cudaMallocManaged 
    // shouldn't be an issue...

    uint32_t* d_A_data,    d_B_data;
    uint32_t* d_A_idxs,    d_B_idxs;
    uint32_t* d_A_idxptrs, d_B_idxptrs;

    uint32_t* d_C_data;
    uint8_t*  d_C_valid;
    uint32_t* h_C_data;
    uint8_t*  h_C_valid;


    // create streams
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // TODO see how multiple streams work
    // cudaStream_t A_data_stream, A_idxs_stream, A_idxptrs_stream;
    // cudaStream_t B_data_stream, B_idxs_stream, B_idxptrs_stream;
    // checkCudaErrors(cudaStreamCreateWithFlags(A_data_stream, cudaStreamNonBlocking));
    // checkCudaErrors(cudaStreamCreateWithFlags(A_idxs_stream, cudaStreamNonBlocking));
    // checkCudaErrors(cudaStreamCreateWithFlags(A_idxptrs_stream, cudaStreamNonBlocking));
    // checkCudaErrors(cudaStreamCreateWithFlags(B_data_stream, cudaStreamNonBlocking));
    // checkCudaErrors(cudaStreamCreateWithFlags(B_idxs_stream, cudaStreamNonBlocking));
    // checkCudaErrors(cudaStreamCreateWithFlags(B_idxptrs_stream, cudaStreamNonBlocking));

    // stream memory out
    // why don't we just allocate these things on the device itself!!!
    // let's not prematurely optimize... There's no way to interlace the reads
    // and device writes, so it doesn't make a difference when we do the data 
    // transfer...
    checkCudaErrors(cudaMemcpyAsync(d_A_idxptrs, A->idxptrs, A->p+1, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_idxs,    A->idxs,    A->k, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_data,    A->data,    A->k*A->m*A->m, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_idxptrs, B->idxptrs, B->p+1, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_idxs,    B->idxs,    B->k, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_data,    B->data,    B->k*B->m*B->m, cudaMemcpyHostToDevice, stream));

    // initialize C matrix on device
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_data),  A->n*A->n*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_valid), A->n*A->n*sizeof(uint8_t)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_C_data),  A->n*A->n*sizeof(uint32_t)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_C_valid), A->n*A->n*sizeof(uint8_t)));
    checkCudaErrors(cudaMemset(reinterpret_cast<void**>(&d_C_valid), 0, A->n*A->n*sizeof(uint8_t)));
    // JUST WHILE TESTING!
    checkCudaErrors(cudaMemset(reinterpret_cast<void**>(&d_C_data), 0, A->n*A->n*sizeof(uint32_t)));

    // assuming A->m = B->m = m
    int p = A->p;
    int m = A->m;
    dim3 threads(m, m);
    dim3 grid(p, p);

    // now multiply the stuff out
    if (A->m == 4) {
        sparse_mul_cuda<4><<<grid, threads, 0, stream>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            d_C_data, d_C_valid, n
        );
    }
    else {
        sparse_mul_cuda<8><<<grid, threads, 0, stream>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            d_C_data, d_C_valid, n
        );
    }

    // TODO compress the C matrix into CSR format
    // for now, we can just copy it over
    checkCudaErrors(cudaMemcpyAsync(d_C_data, h_C_data, n*n*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(d_C_valid, h_C_valid, n*n*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaFree(d_A_idxptrs));
    checkCudaErrors(cudaFree(d_A_idxs));
    checkCudaErrors(cudaFree(d_A_data));
    checkCudaErrors(cudaFree(d_B_idxptrs));
    checkCudaErrors(cudaFree(d_B_idxs));
    checkCudaErrors(cudaFree(d_B_data));
    checkCudaErrors(cudaFree(d_C_data));
    checkCudaErrors(cudaFree(d_C_valid));
    checkCudaErrors(cudaFreeHost(h_C_valid));

    return h_C_data
}

int main(int argc, char** argv) {

    CSRMatrix* A = new CSRMatrix();
    CSRMatrix* B = new CSRMatrix();

    uint32_t* C_data = sparse_matrix_multiply(A, B);

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << C_data[n*i+j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

