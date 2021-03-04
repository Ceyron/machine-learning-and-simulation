#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Sparse_Coordinate {
    size_t n_rows;
    size_t n_cols;
    size_t nnz;
    size_t* row_indices;
    size_t* col_indices;
    double* values;
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
        2,  0,  0,  2,  0,
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

    Sparse_Coordinate A_coo;

    create_sparse_coordinate(A, n_rows, n_cols, nnz, &A_coo);

    print_sparse_coordinate(&A_coo);

    matrix_vector_sparse_coordinate(&A_coo, x, Ax);
    
    printf("\n");
    for (size_t i=0; i<n_cols; ++i) {
        printf("%2.2f\n", Ax[i]);
    }


    free_sparse_coordinate(&A_coo);

    return EXIT_SUCCESS;
}


int create_sparse_coordinate(
    const double* A,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    Sparse_Coordinate* A_coo
) {
    A_coo->n_rows = n_rows;
    A_coo->n_cols = n_cols;
    A_coo->nnz = nnz;
    A_coo->row_indices = calloc(nnz, sizeof(size_t));
    A_coo->col_indices = calloc(nnz, sizeof(size_t));
    A_coo->values = calloc(nnz, sizeof(double));

    size_t nnz_id = 0;

    for (size_t i=0; i<n_rows; ++i) {
        for (size_t j=0; j<n_cols; ++j) {
            if (A[i*n_cols + j] != 0) {
                A_coo->row_indices[nnz_id] = i;
                A_coo->col_indices[nnz_id] = j;
                A_coo->values[nnz_id] = A[i*n_cols + j];
                nnz_id++;
            }
        }
    }

    return EXIT_SUCCESS;
}

int print_sparse_coordinate(const Sparse_Coordinate* A_coo) {
    printf("\n");
    printf("row\tcol\tval\n");
    printf("---\t---\t---\n");
    for(size_t nnz_id=0; nnz_id<A_coo->nnz; ++nnz_id) {
        size_t row_id = A_coo->row_indices[nnz_id];
        size_t col_id = A_coo->col_indices[nnz_id];
        double value = A_coo->values[nnz_id];

        printf("%d\t%d\t%02.2f\n", row_id, col_id, value);
    }

    return EXIT_SUCCESS;
}

int matrix_vector_sparse_coordinate(
    const Sparse_Coordinate* A_coo,
    const double* vec,
    double* res
) {
    for (size_t i=0; i<A_coo->n_cols; ++i) {
        res[i] = 0.0;
    }

    for (size_t nnz_id=0; nnz_id<A_coo->nnz; ++nnz_id) {
        size_t row_id = A_coo->row_indices[nnz_id];
        size_t col_id = A_coo->col_indices[nnz_id];
        double value = A_coo->values[nnz_id];

        res[row_id] += value * vec[col_id];
    }

    return EXIT_SUCCESS;
}

int free_sparse_coordinate(Sparse_Coordinate* A_coo) {
    free(A_coo->row_indices);
    free(A_coo->col_indices);
    free(A_coo->values);

    return EXIT_SUCCESS;
}