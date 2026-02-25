#pragma once

#include "solver.h"

void slave_mpi_task(TLinearSystem lin_sys, int *displs);

SolverStatus master_mpi_task(
    const int *displs,
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);