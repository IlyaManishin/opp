#include "mpi_solvers/mpi_solver.h"
#include "solver.h"

#include "io_utils.h"
#include "matrix.h"

#include <math.h>
#include <stdlib.h>

SolverStatus solve_mpi_impl(
    TLinearSystem linSys,
    double *x,
    int *displs,
    double eps,
    int maxIters,
    int rank,
    int size)
{
    SolverStatus st = SOL_OK;

    if (size == 1)
        return solve_linear_single_impl(linSys, x, eps, maxIters);

    if (rank == 0)
        st = master_mpi_task(linSys, x, displs, eps, maxIters);
    else
        slave_mpi_task(linSys, displs);

    return st;
}