#include <fstream>
#include <iostream>
#include <cstring>
#include <map>

using idx_t = uint64_t;

enum CompressionType {
    CT_ROW, CT_COL
};

// https://matteding.github.io/2019/04/25/sparse-matrices/#compressed-sparse-rowcolumn
struct BCSMatrix {
    int n;
    int m;
    int k;
    CompressionType ct;

    int *idxptrs;
    int *idxs;
    uint32_t *data;

    BCSMatrix(int _n, int _m, int _k, CompressionType _ct); 
    BCSMatrix(char* filename, CompressionType _ct); 
    BCSMatrix(); 

    void dense_print();
    void print();

    void save(char* filename);

    ~BCSMatrix();
};

struct CudaSparseMatrix {

    int n;
    int m;
    int k;
    CompressionType ct;

    uint32_t* data;
    uint8_t* valid;

    CudaSparseMatrix(int _n, int _m);
    ~CudaSparseMatrix();
};
