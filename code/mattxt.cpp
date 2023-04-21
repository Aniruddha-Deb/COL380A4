#include "sparsemat.hpp"
#include <iostream>

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Provide input matrix and element size" << std::endl;
        return 0;
    }

    BCSMatrix *m = new BCSMatrix(argv[1], CT_ROW);
    
    m->dense_print();

    delete m;
    return 0;
}
