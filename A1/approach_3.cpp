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


SparseMatrix* sparse_mat_square(SparseMatrix *mat) {
    
    // TODO need row-wise and column-wise locks here
    int off = mat->m*mat->m;
    // omp_lock_t *i_locks = new omp_lock_t[mat->n/mat->m];
    // omp_lock_t *j_locks = new omp_lock_t[mat->n/mat->m];
    std::unordered_map<int,std::unordered_set<int>> i_map;
    std::unordered_map<int,std::unordered_set<int>> j_map;

    // for (int i=0; i<mat->n/mat->m; i++) {
    //     omp_init_lock(&i_locks[i]);
    //     omp_init_lock(&j_locks[i]);
    // }

    for (int i=0; i<mat->k; i++) {
        i_map[mat->I[i]].insert(i);
        j_map[mat->J[i]].insert(i);
        //std::cout << mat->I[i] << ", " << mat->J[i] << " is at index " << i << std::endl;
    }

    // std::vector<MatrixMultiplicationJob> jobs;

    std::unordered_set<uint64_t> coords;
    // index map (going from (i,j) -> idx)
    omp_lock_t idx_lock;
    omp_init_lock(&idx_lock);
    std::unordered_map<uint64_t,int> idxs;
    std::unordered_map<uint64_t,vector<MatrixMultiplicationJob>> jobs;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int idx = 0;
    // diagonal stuff
    #pragma omp parallel for
    for (int i=0; i<mat->k; i++) {
        int r = mat->I[i];
        int c = mat->J[i];
        int *m = mat->M[i];

        uint64_t hr = hashcoord(r, r);
        uint64_t hc = hashcoord(c, c);

        #pragma omp critical
        {
            if (idxs.find(hr) == idxs.end()) idxs[hr] = idx++;
            jobs[hr].push_back({m, m, false, true});
            if (r != c) {
                if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
                jobs[hc].push_back({m, m, true, false});
            }
        }
    }

    std::chrono::steady_clock::time_point diags = std::chrono::steady_clock::now();
    std::cout << "Diagonals: " << std::chrono::duration_cast<std::chrono::milliseconds>(diags - begin).count() << " [ms]" << std::endl;

    // non-diagonal stuff
    // #pragma omp parallel for shared(idx)
    for (int i=0; i<mat->k; i++) {
        int r = mat->I[i];
        int c = mat->J[i];

        //omp_set_lock(&i_locks[r]);
        //omp_set_lock(&j_locks[c]);
        i_map[r].erase(i);
        j_map[c].erase(i);
        //omp_unset_lock(&j_locks[c]);
        //omp_unset_lock(&i_locks[r]);

        int c1, c2, i1, i2;
        uint64_t hc;

        // omp_set_lock(&i_locks[r]);
        for (int j : i_map[r]) {
            // same x-coordinate
            c1 = mat->J[i]; c2 = mat->J[j];
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }
            
            hc = hashcoord(c1, c2);
            // omp_set_lock(&idx_lock);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], true, false});
            // omp_unset_lock(&idx_lock);
        }
        // omp_unset_lock(&i_locks[r]);

        // omp_set_lock(&j_locks[c]);
        for (int j : j_map[c]) {
            // same y-coordinate
            c1 = mat->I[i]; c2 = mat->I[j];
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); }

            hc = hashcoord(c1, c2);
            // omp_set_lock(&idx_lock);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, true});
            // omp_unset_lock(&idx_lock);
        }
        // omp_unset_lock(&j_locks[c]);

        if (r == c) continue; // no unique transpose to worry about

        // omp_set_lock(&i_locks[c]);
        for (int j : i_map[c]) {
            c1 = mat->I[i]; c2 = mat->J[j];
            if (c == c2) continue;
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c

            hc = hashcoord(c1, c2);
            // omp_set_lock(&idx_lock);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, false});
            // omp_unset_lock(&idx_lock);
        }
        // omp_unset_lock(&i_locks[c]);

        // omp_set_lock(&j_locks[r]);
        for (int j : j_map[r]) {
            c1 = mat->J[i]; c2 = mat->I[j];
            if (r == c2) continue;
            i1 = i; i2 = j;
            if (c1 > c2) { swap(c1, c2); swap(i1, i2); } // r < c

            hc = hashcoord(c1, c2);
            // omp_set_lock(&idx_lock);
            if (idxs.find(hc) == idxs.end()) idxs[hc] = idx++;
            jobs[hc].push_back({mat->M[i1], mat->M[i2], false, false});
            // omp_unset_lock(&idx_lock);
        }
        // omp_unset_lock(&j_locks[r]);
    }

    std::chrono::steady_clock::time_point non_diags = std::chrono::steady_clock::now();
    std::cout << "Non-Diagonals: " << std::chrono::duration_cast<std::chrono::milliseconds>(non_diags - diags).count() << " [ms]" << std::endl;

    int n_blocks = idx;

    std::cout << n_blocks << std::endl;

    // then create the sparse matrix
    SparseMatrix *M = new SparseMatrix(mat->n, mat->m, n_blocks);
    // std::cout << jobs.size() << std::endl;

    // initialize pointers for jobs
    // this would need a coord -> int map
    #pragma omp parallel
    #pragma omp single
    for (auto& p : jobs) {
        #pragma omp task
        {
            uint64_t hc = p.first;
            int r = get_i(hc), c = get_j(hc);
            auto tasks = p.second;

            M->I[idxs[hc]] = r;
            M->J[idxs[hc]] = c;
            for (int i=0; i<tasks.size(); i++) {
                do_task(mat->m, tasks[i].m1, tasks[i].m2, M->M[idxs[hc]], tasks[i].t1, tasks[i].t2);
            }
        }
    }

    std::chrono::steady_clock::time_point job_end = std::chrono::steady_clock::now();
    std::cout << "Jobs: " << std::chrono::duration_cast<std::chrono::milliseconds>(job_end - non_diags).count() << " [ms]" << std::endl;

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // // TODO clipping
    // int n_jobs = blk_jobs.size();
    // #pragma omp parallel
    // #pragma omp single
    // for (int i=0; i<n_jobs; i++) {
    //     #pragma omp task 
    //     {
    //         auto mbj = blk_jobs[i];
    //         int n = mbj->n;
    //         for (int ii=0; ii<mbj->k; ii++) {
    //             for (int i=0; i<n; i++) {
    //                 for (int j=0; j<n; j++) {
    //                     int acc = 0;
    //                     for (int k=0; k<n; k++) {
    //                         acc = Outer(
    //                             acc,
    //                             inner_lt[mbj->M1[ii][get_idx(n, i, k, mbj->T1[ii])]]
    //                                     [mbj->M2[ii][get_idx(n, k, j, mbj->T2[ii])]]
    //                             );
    //                     }
    //                     mbj->M[get_idx(n, i, j, false)] = Outer(acc, mbj->M[get_idx(n, i, j, false)]);
    //                 }
    //             }
    //         }
    //     }
    // }

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

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

