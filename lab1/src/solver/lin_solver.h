#pragma once

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

void slave_task(
    double *A_part,
    int n,
    int rows);

SolverStatus solve_linear_multy_impl(
    const int *slaves_mask,
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);
