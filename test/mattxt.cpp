#include "densemat.hpp"
#include <iostream>

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Provide input matrix and element size" << std::endl;
        return 0;
    }

    DenseMatrix *m = new DenseMatrix(argv[1], std::stoi(argv[2]));
    
    m->print();
    delete m;
    return 0;
}
