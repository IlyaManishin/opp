#pragma once


typedef enum {
    SOL_OK = 0,
    SOL_MAX_ITERS,
    SOL_INPUT_ERR,
    SOL_INVALID
} SolverStatus;

SolverStatus solve_min_residuals(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters
);
