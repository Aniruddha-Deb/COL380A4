#include "densemat.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Need 2 matrices to compare" << std::endl;
        return 1;
    }

    DenseMatrix *m1 = new DenseMatrix(argv[1], 2);
    DenseMatrix *m2 = new DenseMatrix(argv[2], 2);


    if (m1->equalto(m2)) std::cout << "Both matrices are equal" << std::endl;
    else std::cout << "ERROR: Both matrices are unequal" << std::endl;

    delete m1;
    delete m2;
    return 0;
}
