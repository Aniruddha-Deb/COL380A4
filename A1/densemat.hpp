#include <fstream>
#include <vector>
#include <iostream>

struct DenseMatrix {

    int n;
    int m;
    int k;
    int *mat;

    DenseMatrix(int _n, int _m) {
        n = _n;
        m = _m;
        mat = new int[n*n];
    }

    DenseMatrix(char* filename, int nbytes) {
        std::cout << "n_bytes: " << nbytes << std::endl;
        
        std::ifstream input(filename, std::ios_base::binary);
        input.read((char*)&n, 4);
        input.read((char*)&m, 4);
        int k;
        input.read((char*)&k, 4);

        // std::cout << "Initializing array" << std::endl;
        mat = new int[n*n];
        // std::cout << "Initialized array" << std::endl;
        int t = 0;
        for (int ii=0; ii<k; ii++) {
            int i, j;
            input.read((char*)&i, 4);
            input.read((char*)&j, 4);
            int si = i*m, sj = j*m;
            for (int r=0; r<m; r++) {
                for (int c=0; c<m; c++) {
                    // std::cout << "Reading element at " << (si+r) << ", " << sj+c << std::endl;
                    input.read((char*)&t, nbytes);

                    mat[n*(si+r)+(sj+c)] = t;
                    mat[n*(sj+c)+(si+r)] = t;
                }
            }
        }
    }

    bool equalto(DenseMatrix *other) {
        if (other->n != n) return false;
        if (other->m != m) return false;

        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                if (other->mat[n*i+j] != mat[n*i+j]) return false;
            }
        }
        return true;
    }

    void print() {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                std::cout << mat[n*i+j] << " ";
            }
            std::cout << "\n";
        }
    }

    void save(char* filename, int nbytes) {
        // TODO go over individual chunks and naively check if all values in
        // chunk are zero
        // print();
        std::ofstream output(filename, std::ios_base::binary);
        output.write((char*)&n, 4);
        output.write((char*)&m, 4);

        std::vector<std::pair<int,int>> nz_blocks;

        for (int i=0; i<n/m; i++) { // row
            for (int j=i; j<n/m; j++) { // col
                for (int k=0; k<m; k++) { // small row
                    for (int l=0; l<m; l++) { // small col
                        if (mat[n*(i*m+k) + (j*m+l)] != 0) {
                            // not an empty block; add to nz_blocks
                            // std::cout << "Block " << i << ", " << j << " isn't empty, adding\n";
                            nz_blocks.push_back({i,j});
                            goto block_loop;
                        }
                    }
                }
                block_loop:
                    continue;
            }
        }
        int k = nz_blocks.size();
        output.write((char*)&k, 4);

        char buf[nbytes];
        int t;
        for (auto p : nz_blocks) {
            // std::cout << "Writing block " << p.first << ", " << p.second << "\n";
            output.write((char*)&p.first, 4);
            output.write((char*)&p.second, 4);
            for (int k=0; k<m; k++) { // row
                for (int l=0; l<m; l++) { // col
                    // std::cout << mat[n*(m*p.first+k) + (m*p.second+l)];
                    output.write((char*)&mat[n*(m*p.first+k) + (m*p.second+l)], nbytes);
                }
                // std::cout << "\n";
            }
        }
    }

    ~DenseMatrix() {
        delete mat;
    }
};
