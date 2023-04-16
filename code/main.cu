#include "sparsemat.hpp"
#include <cuda_runtime.h>
#include <cassert>

#define idx(i,j) (((idx_t)(i))<<32 | (j));
#define idx_i(idx) (int((idx)>>32));
#define idx_j(idx) (int(idx));

// taken from the NVIDIA CUDA examples
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

// sparse matrix multiplication C = A * B on the device 
// A is BCSR while B is BCSC
template <int m> __global__
void sparse_mul_cuda(uint32_t *A_data, uint32_t *B_data, uint32_t* C_data, uint8_t* C_valid;
        int *A_idxs, int *B_idxs, int *A_idxmap, int *B_idxmap, int n) {


    // this block multiplies C(bx, by) <- sum_i A(bx, i) * B(i, by)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // just return if there's nothing to do
    if (A_idxmap[bx] == A_idxmap[bx+1] && B_idxmap[by] == B_idxmap[by+1]) return;

    __shared__ uint32_t A_buf[m][m];
    __shared__ uint32_t B_buf[m][m];
    __shared__ uint32_t C_buf[m][m]();

    // use a two-pointer algorithm to find out which matrices to multiply first
    // invariants: 
    //     A_idxs[Ap] <= B_idxs[Bp]
    //     A_idxmap[bx] <= Ap < A_idxmap[bx+1]
    //     B_idxmap[by] <= Bp < B_idxmap[by+1]
    int Ap = A_idxmap[bx];
    int Bp = B_idxmap[by];

    uint8_t multiplied = 0;
    for (; Bp < B_idxmap[by+1]; Bp++) {

        while (A_idxs[Ap] < B_idxs[Bp] && Ap < A_idxmap[bx+1]) Ap++;
        if (Ap >= A_idxmap[bx+1]) break;
        if (A_idxs[Ap] > B_idxs[Bp]) continue;

        // all invariants are satisfied here, so we can multiply
        multiplied = 1;

        // load into A_buf and B_buf 
        A_buf[tx][ty] = A_data[A_idxmap[Ap]*m*m + tx*m + ty]
        B_buf[tx][ty] = B_data[B_idxmap[Bp]*m*m + tx*m + ty]
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

SparseMatrix* sparse_matrix_multiply(CSRMatrix *A, CSRMatrix *B) {

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

    uint32_t* d_A_data, d_B_data;
    uint32_t* d_A_idxs, d_B_idxs;
    uint32_t* d_A_idxptrs, d_B_idxptrs;

    uint32_t* d_C_data;
    uint8_t* d_C_valid;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_data), A->k*A->m*A->m*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxs), A->k*sizeof(int)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A_idxptrs), (A->k+1)*sizeof(int)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_data), B->k*B->m*B->m*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxs), B->k*sizeof(int)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B_idxptrs), (B->k+1)*sizeof(int)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_data), A->n*A->n*sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C_valid), A->n*A->n*sizeof(uint8_t)));

    // should we stream the memory instead? The device accesses the memory only once...

    cudaMalloc(
}

int main(int argc, char** argv) {

    

    return 0;
}

