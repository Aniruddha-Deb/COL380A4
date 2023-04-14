#include <fstream>
#include <iostream>
#include <cstring>
#include <map>

using idx_t = uint64_t;

struct CSRMatrix {
    int n;
    int m;
    int k;
    int *I;
    int *J;
    // ASSUMPTION unsigned for now
    uint32_t **M;

    SparseMatrix(int _n, int _m, int _k); 
    SparseMatrix(char* filename); 

    void print();

    void save(char* filename, int nbytes);

    ~SparseMatrix();
};

struct CSCMatrix {

    int n;
    int m;
    int k;

    int *idx;
    int 

struct CUDASparseMatrix {
    int n;
    int m;
    int k;
    int *I;
    int *J;
    // ASSUMPTION unsigned for now
    uint32_t **M;

    SparseMatrix(int _n, int _m, int _k); 
    SparseMatrix(char* filename); 

    void print();

    void save(char* filename, int nbytes);

    ~SparseMatrix();
};

