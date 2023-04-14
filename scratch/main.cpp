#include "sparsemat.hpp"

#define idx(i,j) (((idx_t)(i))<<32 | (j));
#define idx_i(idx) (int((idx)>>32));
#define idx_j(idx) (int(idx));

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


