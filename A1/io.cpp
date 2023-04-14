#include "library.hpp"
#include <iostream>
#include <fstream>

struct DenseMatrix {

    int n;
    int m;
    int k;
    int *mat;

    DenseMatrix(int _n, int _m, int _k) {
        n = _n;
        m = _m;
        k = _k;
        mat = new int[n*n];
    }

    DenseMatrix(char* filename) {
        
        std::ifstream input(filename, std::ios_base::binary);
        input.read((char*)&n, 4);
        input.read((char*)&m, 4);
        input.read((char*)&k, 4);

        mat = new int[n*n];
        char t;
        for (int ii=0; ii<k; ii++) {
            int i, j;
            input.read((char*)&i, 4);
            input.read((char*)&j, 4);
            int si = i*m, sj = j*m;
            for (int r=0; r<m; r++) {
                for (int c=0; c<m; c++) {
                    // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                    input.read(&t, 1);
                    mat[n*(si+r)+(sj+c)] = t;
                    mat[n*(sj+c)+(si+r)] = t;
                }
            }
        }
    }

    void save(char* filename, int nbytes) {
        // TODO go over individual chunks and naively check if all values in
        // chunk are zero
        std::ofstream output(filename, std::ios_base::binary);
        output.write((char*)&n, 4);
        output.write((char*)&m, 4);
        output.write((char*)&k, 4);

        std::vector<std::pair<int,int>> nz_blocks;

        for (int i=0; i<n/m; i++) { // row
            for (int j=i; j<=n/m; j++) { // col
                for (int k=0; k<m; k++) { // small row
                    for (int l=0; l<m; l++) { // small col
                        if (mat[n*(i*m+k) + (j*m+l)] != 0) {
                            // not an empty block; add to nz_blocks
                            nz_blocks.push_back({i,j});
                            goto block_loop;
                        }
                    }
                }
                block_loop:
            }
        }

        for (auto p : nz_blocks) {
            output.write((char*)&p.first, 4);
            output.write((char*)&p.second, 4);
            for (int k=0; k<m; k++) { // row
                for (int l=0; l<m; l++) { // col
                    output.write((char*)&mat[n*(p.first+k) + (p.second*l)], nbytes);
                }
            }
        }
    }

    ~DenseMatrix() {
        delete mat;
    }
};

int* dense_mat_read(char* filename, int& n, int& m, int& k) {
    // reads in row major order

    std::ifstream input(filename, std::ios_base::binary);
    input.read((char*)&n, 4);
    input.read((char*)&m, 4);
    input.read((char*)&k, 4);

    int *mat = new int[n*n];

    char t;
    for (int ii=0; ii<k; ii++) {
        int i, j;
        input.read((char*)&i, 4);
        input.read((char*)&j, 4);
        int si = i*m, sj = j*m;
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                input.read(&t, 1);
                mat[n*(si+r)+(sj+c)] = t;
                mat[n*(sj+c)+(si+r)] = t;
            }
        }
    }

    return *mat;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cout << "Need input filename" << std::endl;
        return 0;
    }

    int n, m;
    int k;
    int *mat;
    // load in input file from argv 
    std::ifstream input(argv[1], std::ios_base::binary);
    input.read((char*)&n, 4);
    input.read((char*)&m, 4);
    input.read((char*)&k, 4);

    mat = new int[n*n];
    
    std::cout << "Initialized matrix of size " << n << " and block size " << m << std::endl;

    char t;
    for (int ii=0; ii<k; ii++) {
        int i, j;
        input.read((char*)&i, 4);
        input.read((char*)&j, 4);
        int si = i*m, sj = j*m;
        for (int r=0; r<m; r++) {
            for (int c=0; c<m; c++) {
                // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                input.read(&t, 1);
                mat[n*(si+r)+(sj+c)] = t;
                mat[n*(sj+c)+(si+r)] = t;
            }
        }
    }

    std::cout << "Read matrix" << std::endl;

    // do the operation

    int *out = new int[n*n];

    for (int i=0; i<n; i++) {
        for (int j=i; j<n; j++) {
            int acc = 0;
            for (int k=0; k<n; k++) {
                acc = Outer(acc, Inner(mat[n*i+k], mat[n*k+j]));
            }
            out[n*i+j] = acc;
            out[n*j+i] = acc;
        }
    }
                
    std::cout << "Input matrix" << std::endl;

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << mat[n*i + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Output matrix" << std::endl;
        
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            std::cout << out[n*i + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
