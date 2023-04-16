#include "sparsemat.hpp"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {

    int t;
    std::cin >> t;
    while (t-- > 0) {

        BCSMatrix *mat = new BCSMatrix();
        std::cout << "Done reading matrix: \n";
        mat->print();
        std::cout << "Dense representation: \n";
        mat->dense_print();

        delete mat;
    }
}
