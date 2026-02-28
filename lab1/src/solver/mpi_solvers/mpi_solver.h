#pragma once

#include "../solver.h"
#include "io_utils.h"

void slave_mpi_task(
    TLinearSystem linSys,
    int *displs);

SolverStatus master_mpi_task(
    TLinearSystem linSys,
    double *x,
    const int *displs,
    double eps,
    int maxIters);