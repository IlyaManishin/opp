#include <stdlib.h>
#include <cblas.h>

#include "matrix.h"

double *matrix_create(int rows, int cols) {
    return (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
}

void matrix_free(double *A) {
    free(A);
}

double matrix_get(const double *A, int cols, int i, int j) {
    return A[i * cols + j];
}

void matrix_set(double *A, int cols, int i, int j, double value) {
    A[i * cols + j] = value;
}

void matrix_swap_rows(double *A, int cols, int r1, int r2) {
    if (r1 == r2) return;

    for (int j = 0; j < cols; j++) {
        double tmp = A[r1 * cols + j];
        A[r1 * cols + j] = A[r2 * cols + j];
        A[r2 * cols + j] = tmp;
    }
}

void matrix_mul_vec(const double *A, int rows, int cols, const double *x, double *y) {
    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows,
        cols,
        1.0,
        A,
        cols,
        x,
        1,
        0.0,
        y,
        1
    );
}
