#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "sparsemat.hpp"

int main(int argc, char** argv) {
    
    // depending on argv[1], make a random matrix with the given no. of nonzero 
    // input blocks

    std::cout << "Starting" << std::endl;

    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    std::cout << "Got numbers" << std::endl;
    
    SparseMatrix *mat = new SparseMatrix(n, m, k);
    int u = n/m;
    int nb = u*(u+1)/2;

    std::random_device dev;
    std::mt19937 rng(dev());

    // choose a random subset of blocks to populate
    std::unordered_set<uint64_t> populated;

    std::uniform_int_distribution<std::mt19937::result_type> row(0,u-1);
    std::uniform_int_distribution<std::mt19937::result_type> col(0,u-1);
    std::uniform_int_distribution<std::mt19937::result_type> datagen(0,15);
    #pragma omp parallel
    #pragma omp single
    for (int i=0; i<k; i++) {

        int r = row(rng);
        int c = col(rng);
        int hc = hashcoord(r,c);
        if (populated.find(hc) != populated.end()) continue;
        populated.insert(hc);
        #pragma omp task
        {
            mat->I[i] = r;
            mat->J[i] = c;
            for (int j=0; j<m*m; j++) {
                mat->M[i][j] = datagen(rng);
            }
        }
    }

    std::cout << "saving matrix" << std::endl;

    mat->save(argv[4],1);

    std::cout << "saved" << std::endl;

    delete mat;

    return 0;
}
