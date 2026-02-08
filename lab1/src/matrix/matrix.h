#pragma once

#include <stddef.h>

double *matrix_create(int rows, int cols);
void matrix_free(double *A);

double matrix_get(const double *A, int cols, int i, int j);
void matrix_set(double *A, int cols, int i, int j, double value);

void matrix_swap_rows(double *A, int cols, int r1, int r2);

void matrix_mul_vec(const double *A, int rows, int cols, const double *x, double *y);