#include <iostream>
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

struct 

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

SparseMatrix* sparse_mat_square(SparseMatrix *mat) {
    
    //std::cout << mat->k << std::endl;

    std::unordered_map<int,std::unordered_set<int>> i_map;
    std::unordered_map<int,std::unordered_set<int>> j_map;

    for (int i=0; i<mat->k; i++) {
        i_map[mat->I[i]].insert(i);
        j_map[mat->J[i]].insert(i);
        //std::cout << mat->I[i] << ", " << mat->J[i] << " is at index " << i << std::endl;
    }

    // std::vector<MatrixMultiplicationJob> jobs;

    std::unordered_set<uint64_t> coords;
    // index map (going from (i,j) -> idx)
    std::unordered_map<uint64_t,int> idxs;
    std::unordered_map<uint64_t,vector<MatrixMultiplicationJob>> jobs;

    // claim: a block can only multiply itself with another block once.
    // so we need to avoid the cases when diagonal elements multiply themselves
    // twice (x1 == x2 == y1). 
    // simple hack would be to ignore cross-equivalences when element is
    // diagonal...
    // yeah, let's do that for now.
    int idx = 0;
    for (int i=0; i<mat->k; i++) {
        // remove itself from the maps so no self-collision occurs
        i_map[mat->I[i]].erase(i);
        j_map[mat->J[i]].erase(i);
        int c1, c2, i1, i2;
        uint64_t hc, hc1, hc2;
        for (int c : i_map[mat->I[i]]) {
            // they have the same x-coordinate
            c1 = mat->J[i]; c2 = mat->J[c];
            i1 = i; i2 = c;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << ")T, (" << mat->I[c] << "," << mat->J[c] << ") to " << c1 << ", " << c2 << std::endl;
            hc = hashcoord(c1, c2);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], true, false});
        }
        for (int c : i_map[mat->J[i]]) {
            // their y and x coordinates match (Ac = Br)
            // if i is on the diagonal, it's x-coord will already match with 
            if (mat->I[i] == mat->J[i] or mat->I[c] == mat->J[c]) break;
            c1 = mat->I[i]; c2 = mat->J[c];
            i1 = i; i2 = c;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << "), (" << mat->I[c] << "," << mat->J[c] << ") to " << c1 << ", " << c2 << std::endl;
            hc = hashcoord(c1, c2);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, false});
        }
        for (int c : j_map[mat->I[i]]) {
            // their x and y coordinates match (Ar = Bc)
            // if i is on the diagonal, it's x-coord will already match with 
            if (mat->I[i] == mat->J[i] or mat->I[c] == mat->J[c]) break;
            c1 = mat->J[i]; c2 = mat->I[c];
            i1 = i; i2 = c;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << "), (" << mat->I[c] << "," << mat->J[c] << ") to " << c1 << ", " << c2 << std::endl;
            hc = hashcoord(c1, c2);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, false});
        }
        // need to check the other way as well, because we remove the matrix
        // from the list after this step
        // so check for x and y matches as well
        for (int c : j_map[mat->J[i]]) {
            // they have the same y-coordinate
            c1 = mat->I[i]; c2 = mat->I[c];
            i1 = i; i2 = c;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << "), (" << mat->I[c] << "," << mat->J[c] << ")T to " << c1 << ", " << c2 << std::endl;
            hc = hashcoord(c1, c2);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, true});
        }

        // generate mat squares on the diagonals
        if (mat->I[i] == mat->J[i]) {
            // elem on the diagonal itself: only one diagonal term will have
            // squared mat
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << "), (" << mat->I[i] << "," << mat->J[i] << ") to " << mat->I[i] << ", " << mat->I[i] << std::endl;
            hc = hashcoord(mat->I[i], mat->I[i]);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i], mat->M[i], false, false});
        }
        else {
            // elem off-diagonal: two diagonal terms will have the squared mat
            // just realized, the transposition here doesn't matter because
            // it will anyway be diagonal
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << "), (" << mat->I[i] << "," << mat->J[i] << ")T to " << mat->I[i] << ", " << mat->I[i] << std::endl;
            // std::cout << "TODO mult (" << mat->I[i] << "," << mat->J[i] << ")T, (" << mat->I[i] << "," << mat->J[i] << ") to " << mat->J[i] << ", " << mat->J[i] << std::endl;
            hc1 = hashcoord(mat->I[i], mat->I[i]);
            if (idxs.find(hc1) == idxs.end()) idxs[hc1] = idx++;
            hc2 = hashcoord(mat->J[i], mat->J[i]);
            if (idxs.find(hc2) == idxs.end()) idxs[hc2] = idx++;
            jobs[hc1].push_back({mat->M[i], mat->M[i], false, true});
            jobs[hc2].push_back({mat->M[i], mat->M[i], true, false});
        }
    }

    int n_blocks = idx;

    std::cout << n_blocks << std::endl;

    // then create the sparse matrix
    SparseMatrix *M = new SparseMatrix(mat->n, mat->m, n_blocks);
    std::vector<MatrixBlockJob*> blk_jobs;
    // std::cout << jobs.size() << std::endl;

    // initialize pointers for jobs
    // this would need a coord -> int map
    for (auto& p : jobs) {
        uint64_t hc = p.first;
        int r = get_i(hc), c = get_j(hc);
        auto tasks = p.second;

        M->I[idxs[hc]] = r;
        M->J[idxs[hc]] = c;
        MatrixBlockJob* mbj = new MatrixBlockJob(mat->m, tasks.size(), M->M[idxs[hc]]);
        for (int i=0; i<tasks.size(); i++) {
            mbj->M1[i] = tasks[i].m1;
            mbj->M2[i] = tasks[i].m2;
            mbj->T1[i] = tasks[i].t1;
            mbj->T2[i] = tasks[i].t2;
        }
        //std::cout << "Job " << r << ", " << c << " to " << idxs[hc] << std::endl;
        blk_jobs.push_back(mbj);
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // TODO clipping
    int n_jobs = blk_jobs.size();
    #pragma omp parallel
    #pragma omp single
    for (int i=0; i<n_jobs; i++) {
        #pragma omp task 
        {
            auto mbj = blk_jobs[i];
            int n = mbj->n;
            for (int ii=0; ii<mbj->k; ii++) {
                for (int i=0; i<n; i++) {
                    for (int j=0; j<n; j++) {
                        int acc = 0;
                        for (int k=0; k<n; k++) {
                            acc = Outer(
                                acc,
                                inner_lt[mbj->M1[ii][get_idx(n, i, k, mbj->T1[ii])]]
                                        [mbj->M2[ii][get_idx(n, k, j, mbj->T2[ii])]]
                                );
                        }
                        mbj->M[get_idx(n, i, j, false)] = Outer(acc, mbj->M[get_idx(n, i, j, false)]);
                    }
                }
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

    // return the sparse matrix
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

            SparseMatrix *ans = sparse_mat_square(mat);

            ans->print();

            delete mat;
            delete ans;
        }
    }
    else {

        char* output_name = "output";
        if (argc >= 3) {
            output_name = argv[2];
        }

        char* infile = argv[1];
        SparseMatrix *m1 = new SparseMatrix(infile);

        SparseMatrix *m2 = sparse_mat_square(m1);

        //std::cout << "Computed m2" << std::endl;

        m2->save(output_name, 2);

        delete m1;
        delete m2;
    }
}

