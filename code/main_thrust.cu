#include "sparsemat.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cassert>
#include <cstdio>
#include <iostream>
// #include <helper_cuda.h>
// #include <helper_functions.h>

// sparse matrix multiplication C = A * B on the device 
// A is BCSR while B is BCSC
template <int m> __global__
void sparse_mul_cuda(uint32_t *A_data, int *A_idxs, int *A_idxptrs, 
        uint32_t *B_data, int *B_idxs, int *B_idxptrs, 
        uint32_t* C_data, uint8_t* C_valid, int n) {


    // this block multiplies C(bx, by) <- sum_i A(bx, i) * B(i, by)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // just return if there's nothing to do
    if (A_idxptrs[bx] == A_idxptrs[bx+1] && B_idxptrs[by] == B_idxptrs[by+1]) return;

    __shared__ uint32_t A_buf[m][m];
    __shared__ uint32_t B_buf[m][m];
    __shared__ uint32_t C_buf[m][m];

    C_buf[tx][ty] = 0;

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
        A_buf[tx][ty] = A_data[A_idxptrs[Ap]*m*m + tx*m + ty];
        B_buf[tx][ty] = B_data[B_idxptrs[Bp]*m*m + tx*m + ty];
        __syncthreads();

        // Do the multiplication
        #pragma unroll
        for (int i=0; i<m; i++) {
            C_buf[tx][ty] += A_buf[tx][i] * B_buf[i][ty];
            // TODO add clipping here?
        }
        __syncthreads();
    }

    if (!multiplied) return;

    // for (int i=0; i<m; i++) {
    //     for (int j=0; j<m; j++) {
    //         printf("%d ", C_buf[i][j]);
    //     }
    //     printf("\n");
    // }

    // copy C_buf to C_data: each thread copies one element
    C_valid[bx*m + by] = 1;
    C_data[(bx*m + tx)*n + by*m + ty] = C_buf[tx][ty];
}

// thrust functor for coordinate transformations
struct idx_to_row {

    const int p;
    idx_to_row(int _p): p(_p) {}

    __host__ __device__
    float operator()(const int& idx) {
        return idx / p;
    }
};

struct idx_to_col {

    const int p;
    idx_to_row(int _p): p(_p) {}

    __host__ __device__
    float operator()(const int& idx) {
        return idx % p;
    }
};

//BCSMatrix* sparse_matrix_multiply(BCSMatrix *A, BCSMatrix *B) {
uint32_t* sparse_matrix_multiply(BCSMatrix *A, BCSMatrix *B) {

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

    int n = A->n;
    int m = A->m;
    int p = A->p;
    thrust::device_vector<uint32_t> C_data(n*n);
    thrust::device_vector<uint8_t> C_valid(p*p, 0);

    uint32_t *d_A_data,    *d_B_data;
    int *d_A_idxs,    *d_B_idxs;
    int *d_A_idxptrs, *d_B_idxptrs;

    // create streams
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxptrs), sizeof(int)*(A->p+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxs),    sizeof(int)*A->k));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_data),    sizeof(uint32_t)*A->k*A->m*A->m));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxptrs), sizeof(int)*(B->p+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxs),    sizeof(int)*B->k));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_data),    sizeof(uint32_t)*B->k*B->m*B->m));

    checkCudaErrors(cudaMemcpyAsync(d_A_idxptrs, A->idxptrs, sizeof(int)*(A->p+1), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_idxs,    A->idxs,    sizeof(int)*A->k, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_data,    A->data,    sizeof(uint32_t)*A->k*A->m*A->m, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_idxptrs, B->idxptrs, sizeof(int)*(B->p+1), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_idxs,    B->idxs,    sizeof(int)*B->k, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B_data,    B->data,    sizeof(uint32_t)*B->k*B->m*B->m, cudaMemcpyHostToDevice, stream));

    dim3 threads(m, m);
    dim3 grid(p, p);

    // checkCudaErrors(cudaStreamSynchronize(stream));

    // now multiply the stuff out
    if (A->m == 4) {
        sparse_mul_cuda<4><<<grid, threads, 0, stream>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            C_data.data(), C_valid.data(), n
        );
    }
    else {
        sparse_mul_cuda<8><<<grid, threads, 0, stream>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            C_data.data(), C_valid.data(), n
        );
    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    // this is slow, slower than if we do it row-wise imo...
    int n_blocks = thrust::reduce(C_valid.begin(), C_valid.end());
    thrust::host_vector<int> idxs(n_blocks);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(n_blocks),
                    C_valid.begin(),
                    idxs.begin(),
                    _1 == 1)

    thrust::host_vector<uint32_t> final_data(n_blocks*m*m);
    thrust::gather(idxs.begin(), idxs.end(), C_data.begin(), C_

    thrust::host_vector<int> rows(n_blocks);
    thrust::device_vector<int> cols(n_blocks);

    // compress



    // TODO compress the C matrix into CSR format
    // for now, we can just copy it over
    checkCudaErrors(cudaMemcpyAsync(h_C_data,  d_C_data,  n*n*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_C_valid, d_C_valid, p*p*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaFree(d_A_idxptrs));
    checkCudaErrors(cudaFree(d_A_idxs));
    checkCudaErrors(cudaFree(d_A_data));
    checkCudaErrors(cudaFree(d_B_idxptrs));
    checkCudaErrors(cudaFree(d_B_idxs));
    checkCudaErrors(cudaFree(d_B_data));
    checkCudaErrors(cudaFree(d_C_data));
    checkCudaErrors(cudaFree(d_C_valid));
    // checkCudaErrors(cudaFreeHost(h_C_valid));

    return h_C_data;
}

int main(int argc, char** argv) {

    BCSMatrix* A = new BCSMatrix(argv[1], CT_ROW);
    BCSMatrix* B = new BCSMatrix(argv[2], CT_COL);

    A->dense_print();
    std::cout << '\n';
    B->dense_print();
    std::cout << '\n';

    uint32_t* C_data = sparse_matrix_multiply(A, B);

    int n = A->n;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << C_data[n*i+j] << " ";
        }
        std::cout << "\n";
    }
    checkCudaErrors(cudaFreeHost(C_data));

    return 0;
}

