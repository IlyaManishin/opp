#include <stdlib.h>
#include <math.h>

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

void matrix_mul_vec(const double *A, int rows, int cols, int v_size, const double *x, double *y)
{
    for (int i = 0; i < rows; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < v_size; j++)
        {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}


void vec_add(const double *u, const double *v, double *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        res[i] = u[i] + v[i];
    }
}

void vec_sub(const double *u, const double *v, double *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        res[i] = u[i] - v[i];
    }
}

double vec_dot(const double *u, const double *v, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += u[i] * v[i];
    }
    return sum;
}

double vec_norm(const double *v, int n)
{
    return sqrt(vec_dot(v, v, n));
}