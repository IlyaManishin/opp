#pragma once

#include "../utils/io_utils.h"

typedef enum
{
    SOL_OK = 0,
    SOL_MAX_ITERS,
    SOL_INPUT_ERR,
    SOL_INVALID
} SolverStatus;

SolverStatus solve_linear_single_impl(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);

void slave_task(TLinearSystem lin_sys, int *displs);

SolverStatus solve_linear_multy_impl(
    const int *displs,
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);
