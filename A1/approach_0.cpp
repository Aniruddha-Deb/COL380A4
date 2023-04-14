#include "densemat.hpp"
#include "library.hpp"

#include <iostream>

DenseMatrix* dense_mat_square(DenseMatrix* m) {

    DenseMatrix* M = new DenseMatrix(m->n, m->m);

    for (int i=0; i<m->n; i++) {
        for (int j=i; j<m->n; j++) {
            int acc = 0;
            bool first = true;
            for (int k=0; k<m->n; k++) {
                if (first) {
                    acc = Inner(m->mat[m->n*i + k], m->mat[m->n*k + j]);
                    first = false;
                    continue;
                }
                acc = Outer(acc, Inner(m->mat[m->n*i + k], m->mat[m->n*k + j]));
            }
            M->mat[m->n*i + j] = acc;
            M->mat[m->n*j + i] = acc; 
        }
    }

    return M;
}

DenseMatrix* read_mat() {
    int n, m;
    cin >> n >> m;
    DenseMatrix *M = new DenseMatrix(n, m);
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> M->mat[n*i+j];
        }
    }
    return M;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        while (true) {
            DenseMatrix *m1 = read_mat();
        
            DenseMatrix *m2 = dense_mat_square(m1);
            m2->print();
            delete m1;
            delete m2;
        }
        return 0;
    }
    
    char* output_name = "output";
    if (argc >= 3) {
        output_name = argv[2];
    }

    DenseMatrix *m1 = new DenseMatrix(argv[1], 1);
    m1->print();
    std::cout << "\n";

    DenseMatrix *m2 = dense_mat_square(m1);
    m2->print();

    m2->save(output_name, 2);

    delete m1;
    delete m2;
    return 0;
}
