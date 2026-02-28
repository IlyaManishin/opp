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
    TLinearSystem linSys,
    double *x,
    double eps,
    int maxIters);

#ifdef MPI

//***********MULTY***********
SolverStatus solve_mpi_impl(
    TLinearSystem linSys,
    double *x,
    int *displs,
    double eps,
    int maxIters,
    int rank,
    int size);

#endif