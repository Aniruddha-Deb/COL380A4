#include "sparsemat.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <vector>
#include <cassert>

void BCSMatrix::__alloc_mem() {
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&idxptrs), (p+1)*sizeof(int)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&idxs), (k)*sizeof(int)));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&data), (k*m*m)*sizeof(uint32_t)));
}

void BCSMatrix::__free_mem() {
    checkCudaErrors(cudaFreeHost(reinterpret_cast<void**>(&idxptrs)));
    checkCudaErrors(cudaFreeHost(reinterpret_cast<void**>(&idxs)));
    checkCudaErrors(cudaFreeHost(reinterpret_cast<void**>(&data)));
}

BCSMatrix::BCSMatrix(int _n, int _m, int _k, CompressionType _ct) {

    n = _n;
    m = _m;
    k = _k;
    assert(n%m == 0);
    p = n/m;
    ct = _ct;

    __alloc_mem();
}

BCSMatrix::BCSMatrix(char* filename, CompressionType _ct) {

    std::ifstream input(filename, std::ios_base::binary);
    input.read((char*)&n, 4);
    input.read((char*)&m, 4);
    input.read((char*)&k, 4);
    assert(n%m == 0);
    p = n/m;

    ct = _ct;

    __alloc_mem();

    std::vector<std::map<int,uint32_t*>> submats(p);

    uint32_t t;
    for (int ii=0; ii<k; ii++) {
        int i,j;
        input.read((char*)&i, 4);
        input.read((char*)&j, 4);
        uint32_t* submat = new uint32_t[m*m];
        if (ct == CT_COL) std::swap(i, j);
        submats[i][j] = submat;
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                input.read((char*)&t, 2);
                submat[m*r+c] = t;
            }
        }
    }

    data = new uint32_t[k*m*m];
    for (int i=0; i<p; i++) {
        idxptrs[i+1] = idxptrs[i];
        for (auto [subidx, submat] : submats[i]) {
            idxs[idxptrs[i+1]] = subidx;
            memcpy(&data[idxptrs[i+1]*m*m], submat, m*m*sizeof(uint32_t));
            delete submat;
            idxptrs[i+1]++;
        }
    }
}

// reads from stdin
BCSMatrix::BCSMatrix() {
    std::string ct_str;
    std::cin >> n >> m >> k;
    assert(n%m == 0);
    p = n/m;
    std::cin >> ct_str;
    ct = CT_ROW;
    if (ct_str == "col") ct = CT_COL;

    __alloc_mem();

    std::vector<std::map<int,uint32_t*>> submats(p);

    uint32_t t;
    for (int ii=0; ii<k; ii++) {
        int i,j;
        std::cin >> i >> j;
        uint32_t* submat = new uint32_t[m*m];
        if (ct == CT_COL) std::swap(i, j);
        submats[i][j] = submat;
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                std::cin >> t;
                submat[m*r+c] = t;
            }
        }
    }

    for (int i=0; i<p; i++) {
        idxptrs[i+1] = idxptrs[i];
        for (auto [subidx, submat] : submats[i]) {
            idxs[idxptrs[i+1]] = subidx;
            memcpy(&data[idxptrs[i+1]*m*m], submat, m*m*sizeof(uint32_t));
            delete submat;
            idxptrs[i+1]++;
        }
    }
}

void BCSMatrix::print() {

    std::cout << "idxptrs:\n";
    for (int i=0; i<p+1; i++) {
        std::cout << idxptrs[i] << " ";
    }
    std::cout << "\n\nidxs:\n";
    for (int i=0; i<k; i++) {
        std::cout << idxs[i] << " ";
    }
    std::cout << "\n\ndata:\n";
    for (int s=0; s<m; s++) {
        for (int i=0; i<k; i++) {
            for (int j=0; j<m; j++) {
                std::cout << data[i*m*m+s*m+j] << " ";
            }
            std::cout << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void BCSMatrix::dense_print() {
    
    // just convert to a dense matrix
    uint32_t *mat = new uint32_t[n*n]();
    for (int r=0; r<p; r++) {
        for (int cp=idxptrs[r]; cp<idxptrs[r+1]; cp++) {
            for (int br=0; br<m; br++) {
                if (ct == CT_ROW) {
                    memcpy(&mat[r*m*n + br*n + idxs[cp]*m], &data[cp*m*m + br*m], m*sizeof(uint32_t));
                }
                else {
                    memcpy(&mat[idxs[cp]*m*n + br*n + r*m], &data[cp*m*m + br*m], m*sizeof(uint32_t));
                }
            }
        }
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << mat[n*i+j] << " ";
        }
        std::cout << std::endl;
    }
}

void BCSMatrix::save(char* filename) {

    std::ofstream output(filename, std::ios_base::binary);
    output.write((char*)&n, 4);
    output.write((char*)&m, 4);
    output.write((char*)&k, 4);

    for (int r=0; r<p; r++) {
        for (int cp=idxptrs[r]; cp<idxptrs[r+1]; cp++) {
            output.write((char*)&r, 4);
            output.write((char*)&idxs[cp], 4);
            output.write((char*)&data[cp], 4*m*m);
        }
    }
}

BCSMatrix::~BCSMatrix() {
    __free_mem();
}
