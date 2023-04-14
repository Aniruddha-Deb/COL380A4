#include <fstream>
#include <iostream>
#include <cstring>
#include <omp.h>

struct SparseMatrix {
    int n;
    int m;
    int k;
    int *I;
    int *J;
    int *M;

    SparseMatrix(int _n, int _m, int _k) {
        n = _n;
        m = _m;
        k = _k;
        I = new int[k];
        J = new int[k];
        M = new int[m*m*k]{0};
    }

    SparseMatrix(char* filename) {

        std::ifstream input(filename, std::ios_base::binary);
        input.read((char*)&n, 4);
        input.read((char*)&m, 4);
        input.read((char*)&k, 4);

        I = new int[k];
        J = new int[k];
        M = new int[m*m*k];
    
        char t;
        for (int ii=0; ii<k; ii++) {
            input.read((char*)&I[ii], 4);
            input.read((char*)&J[ii], 4);
            for (int r=0; r<m; r++) {
                for (int c=0; c<m; c++) {
                    // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                    input.read(&t, 1);
                    M[m*m*ii+m*r+c] = t;
                }
            }
        }
    }

    void print() {
        // convert to a dense matrix and print
        int *mat = new int[n*n]{0};
        for (int b=0; b<k; b++) {
            for (int r=0; r<m; r++) {
                memcpy(&mat[n*(I[b]*m+r) + (J[b]*m)], &M[m*m*b+m*r], sizeof(int)*m);
            }
        }
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                std::cout << mat[n*i+j] << " ";
            }
            std::cout << std::endl;
        }
        delete mat;
    }

    void save(char* filename, int nbytes) {

        std::ofstream output(filename, std::ios_base::binary);
        output.write((char*)&n, 4);
        output.write((char*)&m, 4);
        output.write((char*)&k, 4);

        for (int ii=0; ii<k; ii++) {
            // first check if the block is empty
            // if (M[ii][0] == 0 and memcmp(&M[ii][0], &M[ii][1], m*m)) {
            //     std::cout << "Encountered empty block: continuing\n";
            //     // lmao, how do we do this if we've already printed k! dum dum
            //     continue;
            // }
            // std::cout << I[ii] << " " << J[ii] << std::endl;
            output.write((char*)&I[ii], 4);
            output.write((char*)&J[ii], 4);
            for (int r=0; r<m; r++) {
                for (int c=0; c<m; c++) {
                    // TODO truncation, but that's ok for now
                    output.write((char*)&M[m*m*ii+m*r+c], nbytes);
                }
            }
        }
    }

    ~SparseMatrix() {
        delete I;
        delete J;
        delete M;
    }
};

inline uint64_t hashcoord(int i, int j) {
    return ((uint64_t)i)<<32 | j;
}

inline int get_i(uint64_t hash) {
    return (hash&(((uint64_t)0xFFFFFFFF)<<32))>>32;
}

inline int get_j(uint64_t hash) {
    return (int)hash;
}

int add_fn(int i, int j) {
    return i+j;
}
