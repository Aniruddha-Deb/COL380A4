#include <iostream>
#include "library.hpp"
#include "sparsemat.hpp"

#include <map>
#include <chrono>

static short inner_lt[256][256];
static short outer_lt[256][256];

inline int get_idx(int n, int i, int j, bool t) {
    return t ? (n*j+i) : (n*i+j);
}

struct MatrixMultiplicationJob {
    int *m1, *m2;
    bool t1, t2;
};

struct MatrixBlockJob {
    int n, k;
    int **M1, **M2;
    int *M;
    bool *T1, *T2;

    MatrixBlockJob(int _n, int _k, int *_M) {
        n = _n;
        k = _k;
        M = _M;
        M1 = new int*[k];
        M2 = new int*[k];
        T1 = new bool[k];
        T2 = new bool[k];
    }

    ~MatrixBlockJob() {
        delete M1, M2, T1, T2;
    }
        
};

inline void swap(int& i, int& j) {
    int t = i; i = j; j = t;
}

void do_task(int m, int *m1, int *m2, int *M, bool t1, bool t2) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) {
            int acc = 0;
            for (int k=0; k<m; k++) {
                acc = Outer(
                    acc,
                    inner_lt[m1[get_idx(m, i, k, t1)]][m2[get_idx(m, k, j, t2)]]
                );
            }
            M[get_idx(m, i, j, false)] = Outer(acc, M[get_idx(m, i, j, false)]);
        }
    }
}

DynamicSparseMatrix* sparse_mat_square(SparseMatrix *mat) {
    
    //std::cout << mat->k << std::endl;

    std::unordered_map<int,std::unordered_set<int>> i_map;
    std::unordered_map<int,std::unordered_set<int>> j_map;

    for (int i=0; i<mat->k; i++) {
        i_map[mat->I[i]].insert(i);
        j_map[mat->J[i]].insert(i);
        //std::cout << mat->I[i] << ", " << mat->J[i] << " is at index " << i << std::endl;
    }

    DynamicSparseMatrix *M = new DynamicSparseMatrix(mat->n, mat->m);
    
    // do all the diagonal stuff first
    #pragma omp parallel
    #pragma omp single
    for (int i=0; i<mat->k; i++) {
        int r = mat->I[i];
        int c = mat->J[i];
        int *m = mat->M[i];

        int *m1 = M->get(r,r);
        // std::cout << "Making Task 1" << std::endl;
        // #pragma omp task depend(inout: *m1)
        #pragma omp task depend(inout: *m1)
        do_task(mat->m, m, m, m1, false, true);

        if (r == c) {
            continue;
        }

        int *m2 = M->get(c,c);
        // std::cout << "Making Task 2" << std::endl;
        #pragma omp task depend(inout: *m2)
        do_task(mat->m, m, m, m2, true, false);
    }

    // now non-diagonal stuff
    #pragma omp parallel
    #pragma omp single
    for (int i=0; i<mat->k; i++) {
        int r = mat->I[i];
        int c = mat->J[i];
        i_map[r].erase(i);
        j_map[c].erase(i);
        int c1, c2, i1, i2;
        uint64_t hc, hc1, hc2;
        for (int j : i_map[r]) {
            // same x-coordinate
            c1 = mat->J[i]; c2 = mat->J[j];
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }
            
            int *m1 = M->get(c1, c2);
            #pragma omp task depend(inout: *m1)
            do_task(mat->m, mat->M[i1], mat->M[i2], m1, true, false);
        }
        for (int j : j_map[c]) {
            // same y-coordinate
            c1 = mat->I[i]; c2 = mat->I[j];
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }

            int *m1 = M->get(c1, c2);
            #pragma omp task depend(inout: *m1)
            do_task(mat->m, mat->M[i1], mat->M[i2], m1, false, true);
        }

        if (r == c) continue; // no unique transpose to worry about

        for (int j : i_map[c]) {
            c1 = mat->I[i]; c2 = mat->J[j];
            if (c == c2) continue;
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c

            int *m1 = M->get(c1, c2);
            #pragma omp task depend(inout: *m1)
            do_task(mat->m, mat->M[i1], mat->M[i2], m1, false, false);
        }
        for (int j : j_map[r]) {
            c1 = mat->J[i]; c2 = mat->I[j];
            if (r == c2) continue;
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c

            int *m1 = M->get(c1, c2);
            #pragma omp task depend(inout: *m1)
            do_task(mat->m, mat->M[i1], mat->M[i2], m1, false, false);
        }
    }

    std::cout << M->k << std::endl;

    return M;
}

int main(int argc, char** argv) {

    // populate lt
    for (int i=0; i<256; i++) {
        for (int j=i; j<256; j++) {
            inner_lt[i][j] = inner_lt[j][i] = Inner(i,j);
            //outer_lt[i][j] = outer_lt[j][i] = Outer(i,j);
        }
    }

    if (argc < 2) {
        // we're in test mode
        int t;
        std::cin >> t;
        while (t-- > 0) {
            int n, m, k;
            std::cin >> n >> m >> k;
            SparseMatrix *mat = new SparseMatrix(n, m, k);
            for (int c=0; c<k; c++) {
                std::cin >> mat->I[c] >> mat->J[c];
                for (int i=0; i<m; i++) {
                    for (int j=0; j<m; j++) {
                        std::cin >> mat->M[c][m*i+j];
                    }
                }
            }

            auto ans = sparse_mat_square(mat);

            auto sm = ans->to_sparse_mat();
            sm->print();

            delete mat;
            delete ans;
            delete sm;
        }
    }
    else {

        char* output_name = "output";
        if (argc >= 3) {
            output_name = argv[2];
        }

        char* infile = argv[1];
        SparseMatrix *m1 = new SparseMatrix(infile);

        auto m2 = sparse_mat_square(m1);

        m2->save(output_name, 2);

        delete m1;
        delete m2;
    }
}

