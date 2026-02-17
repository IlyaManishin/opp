#include <cblas.h>
#include <stdlib.h>

#include "matrix.h"

double *matrix_create(int rows, int cols)
{
    return (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
}

double *vector_create(int n)
{
    return (double *)calloc((size_t)n, sizeof(double));
}

void vector_free(double *vec)
{
    free(vec);
}

void matrix_free(double *A)
{
    free(A);
}

void matrix_mul_vec(const double *A, int rows, int cols, const double *x, double *y)
{
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
        1);
}

double vec_dot(const double *u, const double *v, int n)
{
    return cblas_ddot(n, u, 1, v, 1);
}