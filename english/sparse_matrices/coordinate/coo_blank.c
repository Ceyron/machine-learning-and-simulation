#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Sparse_Coordinate {

} Sparse_Coordinate;


int create_sparse_coordinate(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    Sparse_Coordinate* A_coo
);

int print_sparse_coordinate(const Sparse_Coordinate* A_coo);

int matrix_vector_sparse_coordinate(
    const Sparse_Coordinate* A_coo,
    const double* vec,
    double* res
);

int free_sparse_coordinate(Sparse_Coordinate* A_coo);


int main (int argc, char** argv) {
    size_t n_rows = 5;
    size_t n_cols = 5;
    size_t nnz = 12;

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


int create_sparse_coordinate(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    Sparse_Coordinate* A_coo
) {
    return EXIT_SUCCESS;
}

int print_sparse_coordinate(const Sparse_Coordinate* A_coo) {
    return EXIT_SUCCESS;
}

int matrix_vector_sparse_coordinate(
    const Sparse_Coordinate* A_coo,
    const double* vec,
    double* res
) {
    return EXIT_SUCCESS;
}

int free_sparse_coordinate(Sparse_Coordinate* A_coo) {
    return EXIT_SUCCESS;
}