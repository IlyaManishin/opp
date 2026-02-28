#pragma once

#include <stddef.h>

double *matrix_create(int rows, int cols);
void matrix_free(double *A);

double *vector_create(int n);
void vector_free(double *vec);

void matrix_mul_vec(const double *A, int rows, int cols, int v_size, const double *x, double *y);

void vec_add(const double *u, const double *v, double *res, int n);
double vec_dot(const double *u, const double *v, int n);
double vec_norm(const double *v, int n);