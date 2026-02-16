#pragma once

#include <stddef.h>

double *matrix_create(int rows, int cols);
void matrix_free(double *A);

double *vector_create(int n);
void vector_free(double* vec);

void matrix_mul_vec(const double *A, int rows, int cols, const double *x, double *y);