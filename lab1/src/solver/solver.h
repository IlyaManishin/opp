#pragma once

#include "io_utils.h"

typedef enum
{
    SOL_OK = 0,
    SOL_MAX_ITERS,
    SOL_INPUT_ERR,
    SOL_INVALID
} SolverStatus;

//***********SIGNLE***********
SolverStatus solve_linear_single_impl(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);

#ifdef MPI

//***********MULTY***********
SolverStatus solve_mpi_impl(
    TLinearSystem lin_sys,
    double *x,
    int *displs,
    double eps,
    int max_iters,
    int rank,
    int size);

#endif