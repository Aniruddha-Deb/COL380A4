#include "sparsemat.hpp"

int main() {

    // write to testmat.mat

    SparseMatrix *mat = new SparseMatrix(4, 2, 2);

    mat->I[0] = 0;
    mat->J[0] = 0;
    mat->M[0][0] = 1;
    mat->M[0][1] = 2;
    mat->M[0][2] = 2;
    mat->M[0][3] = 3;

    mat->I[1] = 0;
    mat->J[1] = 1;
    mat->M[1][0] = 4;
    mat->M[1][1] = 5;
    mat->M[1][2] = 6;
    mat->M[1][3] = 7;

    mat->save("testmat.mat", 1);

    return 0;
}
