#include "densemat.hpp"
#include <iostream>


int main(int argc, char** argv) {

    int n=4, m=2;
    DenseMatrix *M = new DenseMatrix(4,2);

    for (int i=0; i<n; i++) {
        for (int j=i; j<n; j++) {
            M->mat[n*i+j] = n*i+j;
        }
    }

    M->save("dm_test_1", 1);
    M->save("dm_test_2", 2);
    M->save("dm_test_4", 4);

    return 0;
}
