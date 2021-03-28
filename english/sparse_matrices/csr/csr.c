#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Sparse_CSR {
    size_t n_rows;
    size_t n_cols;
    size_t n_nz;
    size_t* row_ptrs;
    size_t* col_indices;
    double* values;
} Sparse_CSR;


int create_sparse_csr(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t n_nz,
    Sparse_CSR* A_csr
);

int print_sparse_csr(const Sparse_CSR* A_csr);

int matrix_vector_sparse_csr(
    const Sparse_CSR* A_coo,
    const double* vec,
    double* res
);

int free_sparse_csr(Sparse_CSR* A_csr);


int main (int argc, char** argv) {
    size_t n_rows = 5;
    size_t n_cols = 5;
    size_t n_nz = 12;

    double A[] = {
        1,  0,  0,  2,  0,
        3,  4,  2,  5,  0,
        5,  0,  0,  8, 17,
        0,  0, 10, 16,  0,
        0,  0,  0,   0, 14
    };
    double x[] = {
        1,
        2,
        3,
        4,
        5
    };
    double Ax[5];

    // Starts here
    Sparse_CSR A_csr;

    create_sparse_csr(A, n_rows, n_cols, n_nz, &A_csr);

    print_sparse_csr(&A_csr);

    matrix_vector_sparse_csr(&A_csr, x, Ax);

    for (size_t i=0; i<n_rows; ++i) {
        printf("%02.2f\n", Ax[i]);
    }

    free_sparse_csr(&A_csr);

    return EXIT_SUCCESS;
}


int create_sparse_csr(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t n_nz,
    Sparse_CSR* A_csr
) {
    A_csr->n_rows = n_rows;
    A_csr->n_cols = n_cols;
    A_csr->n_nz = n_nz;
    A_csr->row_ptrs = calloc(n_rows+1, sizeof(size_t));
    A_csr->col_indices = calloc(n_nz, sizeof(size_t));
    A_csr->values = calloc(n_nz, sizeof(double));

    size_t nz_id = 0;

    for (size_t i=0; i<n_rows; ++i) {
        A_csr->row_ptrs[i] = nz_id;
        for (size_t j=0; j<n_cols; ++j) {
            if (A[i*n_cols + j] != 0.0) {
                A_csr->col_indices[nz_id] = j;
                A_csr->values[nz_id] = A[i*n_cols + j];
                nz_id++;
            }
        }
    }

    A_csr->row_ptrs[n_rows] = nz_id;

    return EXIT_SUCCESS;
}

int print_sparse_csr(const Sparse_CSR* A_csr) {
    printf("row\tcol\tval\n");
    printf("----\n");
    for (size_t i=0; i<A_csr->n_rows; ++i) {
        size_t nz_start = A_csr->row_ptrs[i];
        size_t nz_end = A_csr->row_ptrs[i+1];
        for (size_t nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            size_t j = A_csr->col_indices[nz_id];
            double val = A_csr->values[nz_id];
            printf("%d\t%d\t%02.2f\n", i, j, val);
        }
    }
    return EXIT_SUCCESS;
}

int matrix_vector_sparse_csr(
    const Sparse_CSR* A_csr,
    const double* vec,
    double* res
) {
    for (size_t i=0; i<A_csr->n_rows; ++i) {
        res[i] = 0.0;
        size_t nz_start = A_csr->row_ptrs[i];
        size_t nz_end = A_csr->row_ptrs[i+1];
        for (size_t nz_id=nz_start; nz_id<nz_end; ++nz_id) {
            size_t j = A_csr->col_indices[nz_id];
            double val = A_csr->values[nz_id];
            res[i] = res[i] + val * vec[j];
        }
    }
    return EXIT_SUCCESS;
}

int free_sparse_csr(Sparse_CSR* A_csr) {
    free(A_csr->row_ptrs);
    free(A_csr->col_indices);
    free(A_csr->values);

    return EXIT_SUCCESS;
}