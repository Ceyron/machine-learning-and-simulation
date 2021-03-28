#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Sparse_CSR {

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

    return EXIT_SUCCESS;
}


int create_sparse_csr(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t n_nz,
    Sparse_CSR* A_csr
) {
    return EXIT_SUCCESS;
}

int print_sparse_csr(const Sparse_CSR* A_csr) {
    return EXIT_SUCCESS;
}

int matrix_vector_sparse_csr(
    const Sparse_CSR* A_csr,
    const double* vec,
    double* res
) {
    return EXIT_SUCCESS;
}

int free_sparse_csr(Sparse_CSR* A_csr) {
    return EXIT_SUCCESS;
}