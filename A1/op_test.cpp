#include "library.hpp"

int main(int argc, char** argv) {

    int i1, i2;
    cin >> i1 >> i2;

    while (i1 >= 0 and i2 >= 0) {
        std::cout << "Inner: " << Inner(i1, i2) << std::endl;
        std::cout << "Outer: " << Outer(i1, i2) << std::endl;
        cin >> i1 >> i2;

    }

    return 0;
}
