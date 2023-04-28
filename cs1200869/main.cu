#include "sparsemat.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cassert>
#include <cstdio>
#include <iostream>
// #include <helper_cuda.h>
// #include <helper_functions.h>

__device__ uint32_t MAX = ~0;

// sparse matrix multiplication C = A * B on the device 
// A is BCSR while B is BCSC
template <int m> __global__
void sparse_mul_cuda(uint32_t *A_data, int *A_idxs, int *A_idxptrs, 
        uint32_t *B_data, int *B_idxs, int *B_idxptrs, 
        uint32_t* C_data, uint8_t* C_valid, int n, int p) {


    // this block multiplies C(bx, by) <- sum_i A(bx, i) * B(i, by)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // just return if there's nothing to do
    if (A_idxptrs[bx] == A_idxptrs[bx+1] || B_idxptrs[by] == B_idxptrs[by+1]) return;

    __shared__ uint32_t A_buf[m][m];
    __shared__ uint32_t B_buf[m][m];
    __shared__ uint64_t C_buf[m][m];

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
        A_buf[tx][ty] = A_data[Ap*m*m + tx*m + ty];
        B_buf[tx][ty] = B_data[Bp*m*m + tx*m + ty];

        __syncthreads();
        // Do the multiplication
        #pragma unroll
        for (int i=0; i<m; i++) {
            C_buf[tx][ty] += (uint64_t)(A_buf[tx][i] * B_buf[i][ty]);
        }
    }

    if (!multiplied) return;

    // for (int i=0; i<m; i++) {
    //     for (int j=0; j<m; j++) {
    //         printf("%d ", C_buf[i][j]);
    //     }
    //     printf("\n");
    // }

    // copy C_buf to C_data: each thread copies one element
    if (tx == 0 && ty == 0) {
        // printf("Multiplied %d,%d\n", bx, by);
        C_valid[bx*p + by] = 1;
    }
    __syncthreads();
    if (C_buf[tx][ty] >> 32) C_data[(bx*m+tx)*n + by*m + ty] = MAX;
    else C_data[(bx*m + tx)*n + by*m + ty] = (uint32_t)C_buf[tx][ty];
}

__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__
void marginalize_rows(uint8_t* C_valid, int* C_rowsums, int p) {

    int rid = blockIdx.x;
    int tid = threadIdx.x;
    // block has 256 threads
    __shared__ int mergesums[256];
    mergesums[tid] = 0;

    // reduce first to a 256-len array
    int i = tid;
    while (i < p) {
        mergesums[tid] += (int)C_valid[rid*p + i];
        i += 256;
    }
    __syncthreads();

    // now reduce mergesums
    if (tid < 128) mergesums[tid] += mergesums[tid+128];
    __syncthreads();
    if (tid < 64) mergesums[tid] += mergesums[tid+64];
    __syncthreads();

    if (tid < 32) warpReduce(mergesums, tid);
    __syncthreads();

    if (tid == 0) {
        C_rowsums[rid+1] = mergesums[0];
    }
}

__global__
void compress_data(uint8_t* C_valid, int* C_idxptrs, int* C_idxs, 
        uint32_t* C_data, uint32_t* C_bcsr_data, int n, int m, int p) {

    int row = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int idx;
    
    if (tx == 0 && ty == 0) idx = C_idxptrs[row];
    __syncthreads();

    for (int i=0; i<p; i++) {
        if (C_valid[p*row + i] == 1) {
            // all threads compress C_data
            C_bcsr_data[idx*m*m + tx*m + ty] = C_data[row*n*m + tx*n + i*m + ty];
            __syncthreads();
            if (tx == 0 && ty == 0) {
                C_idxs[idx] = i;
                idx++;
            }
            __syncthreads();
        }
    }
}

BCSMatrix* sparse_matrix_multiply(BCSMatrix *A, BCSMatrix *B) {

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

    uint32_t *d_A_data,    *d_B_data;
    int *d_A_idxs,    *d_B_idxs;
    int *d_A_idxptrs, *d_B_idxptrs;

    uint32_t* d_C_data;
    int* d_C_idxptrs;
    uint8_t*  d_C_valid;
    //uint32_t* h_C_data;


    // create streams
    // cudaStream_t stream;
    // checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // assuming A->m = B->m = m
    int p = A->p;
    int m = A->m;
    int n = A->n;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxptrs), sizeof(int)*(A->p+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxs),    sizeof(int)*A->k));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_data),    sizeof(uint32_t)*A->k*A->m*A->m));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxptrs), sizeof(int)*(B->p+1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxs),    sizeof(int)*B->k));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_data),    sizeof(uint32_t)*B->k*B->m*B->m));

    // checkCudaErrors(cudaMemcpyAsync(d_A_idxptrs, A->idxptrs, sizeof(int)*(A->p+1), cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_A_idxs,    A->idxs,    sizeof(int)*A->k, cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_A_data,    A->data,    sizeof(uint32_t)*A->k*A->m*A->m, cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_B_idxptrs, B->idxptrs, sizeof(int)*(B->p+1), cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_B_idxs,    B->idxs,    sizeof(int)*B->k, cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_B_data,    B->data,    sizeof(uint32_t)*B->k*B->m*B->m, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemcpy(d_A_idxptrs, A->idxptrs, sizeof(int)*(A->p+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_idxs,    A->idxs,    sizeof(int)*A->k, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_data,    A->data,    sizeof(uint32_t)*A->k*A->m*A->m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B_idxptrs, B->idxptrs, sizeof(int)*(B->p+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B_idxs,    B->idxs,    sizeof(int)*B->k, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B_data,    B->data,    sizeof(uint32_t)*B->k*B->m*B->m, cudaMemcpyHostToDevice));

    // initialize C matrix on device
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_data),  n*n*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_valid), p*p*sizeof(uint8_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_idxptrs), (p+1)*sizeof(int)));
    // checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_C_data),  n*n*sizeof(uint32_t)));
    // JUST WHILE TESTING!
    // checkCudaErrors(cudaMemset(reinterpret_cast<void*>(d_C_data), 0, n*n*sizeof(uint32_t)));
    checkCudaErrors(cudaMemset(reinterpret_cast<void*>(d_C_valid), 0, p*p*sizeof(uint8_t)));

    dim3 threads(m, m);
    dim3 grid(p, p);

    // checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaDeviceSynchronize());

    // now multiply the stuff out
    if (A->m == 4) {
        sparse_mul_cuda<4><<<grid, threads>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            d_C_data, d_C_valid, n, p
        );
    }
    else {
        sparse_mul_cuda<8><<<grid, threads>>>(
            d_A_data, d_A_idxs, d_A_idxptrs, d_B_data, d_B_idxs, d_B_idxptrs,
            d_C_data, d_C_valid, n, p
        );
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_A_idxptrs));
    checkCudaErrors(cudaFree(d_A_idxs));
    checkCudaErrors(cudaFree(d_A_data));
    checkCudaErrors(cudaFree(d_B_idxptrs));
    checkCudaErrors(cudaFree(d_B_idxs));
    checkCudaErrors(cudaFree(d_B_data));

    // checkCudaErrors(cudaStreamSynchronize(stream));
    // checkCudaErrors(cudaMemcpyAsync(h_C_data,  d_C_data,  n*n*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // for (int i=0; i<n; i++) {
    //     for (int j=0; j<n; j++) {
    //         std::cout << h_C_data[i*n+j] << ' ';
    //     }
    //     std::cout << '\n';
    // }

    // uint8_t *h_C_valid;
    // checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_C_valid), p*p*sizeof(uint8_t)));
    // checkCudaErrors(cudaMemcpy(h_C_valid, d_C_valid, p*p*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // for (int i=0; i<p; i++) {
    //     for (int j=0; j<p; j++) {
    //         std::cout << (int)h_C_valid[i*p+j] << ' ';
    //     }
    //     std::cout << '\n';
    // }

    // marginalize the rows out
    marginalize_rows<<<p,256>>>(d_C_valid, d_C_idxptrs, p);
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // sum them up inplace to get the start pointers
    thrust::device_ptr<int> thrust_d_C_idxptrs(d_C_idxptrs);
    thrust::inclusive_scan(thrust_d_C_idxptrs, thrust_d_C_idxptrs+p+1, thrust_d_C_idxptrs);

    int *C_idxptrs;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&C_idxptrs), (p+1)*sizeof(int)));
    checkCudaErrors(cudaMemcpy(C_idxptrs, d_C_idxptrs, (p+1)*sizeof(int), cudaMemcpyDeviceToHost));

    // allocate memory for obtaining row indexes from the last value of d_C_idxptrs
    int *C_idxs, *d_C_idxs;
    uint32_t *C_bcsr_data;
    uint32_t *d_C_bcsr_data;

    int k = C_idxptrs[p];
    
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&C_idxs), k*sizeof(int)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&C_bcsr_data), k*m*m*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_bcsr_data), k*m*m*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_idxs), k*sizeof(int)));

    compress_data<<<p, threads>>>(d_C_valid, d_C_idxptrs, d_C_idxs, d_C_data, d_C_bcsr_data, n, m, p);
    // checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaMemcpy(C_idxs, d_C_idxs, k*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(C_bcsr_data, d_C_bcsr_data, k*m*m*sizeof(int), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpyAsync(C_idxs, d_C_idxs, k*sizeof(int), cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaMemcpyAsync(C_bcsr_data, d_C_bcsr_data, k*m*m*sizeof(int), cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaDeviceSynchronize());

    // TODO compress the C matrix into CSR format
    // for now, we can just copy it over

    checkCudaErrors(cudaFree(d_C_idxptrs));
    checkCudaErrors(cudaFree(d_C_idxs));
    checkCudaErrors(cudaFree(d_C_data));
    checkCudaErrors(cudaFree(d_C_bcsr_data));
    checkCudaErrors(cudaFree(d_C_valid));
    // checkCudaErrors(cudaFreeHost(h_C_valid));

    BCSMatrix *C = new BCSMatrix(n, m, k, CT_ROW);
    C->idxptrs = C_idxptrs;
    C->idxs = C_idxs;
    C->data = C_bcsr_data;

    return C;
}

int main(int argc, char** argv) {

    BCSMatrix* A = new BCSMatrix(argv[1], CT_ROW);
    BCSMatrix* B = new BCSMatrix(argv[2], CT_COL);

    // A->print();
    // A->dense_print();
    // std::cout << '\n';
    // B->print();
    // B->dense_print();
    // std::cout << '\n';

    BCSMatrix *C = sparse_matrix_multiply(A, B);

    checkCudaErrors(cudaDeviceSynchronize());

    // C->print();
    // C->dense_print();
    // std::cout << '\n';

    C->save(argv[3]);

    delete A;
    delete B;
    delete C;

    return 0;
}

