#include "solver.h"
#include "mpi_solvers/mpi_solver.h"

#include "io_utils.h"
#include "matrix.h"

#include <math.h>
#include <stdlib.h>

SolverStatus solve_mpi_impl(
    TLinearSystem lin_sys,
    double *x,
    int *displs,
    double eps,
    int max_iters,
    int rank,
    int size)
{
    SolverStatus st = SOL_OK;

    double *A = lin_sys.A;
    double *b = lin_sys.b;
    int n = lin_sys.n;

    if (size == 1)
        return solve_linear_single_impl(A, n, b, x, eps, max_iters);

    if (rank == 0)
        st = master_mpi_task(displs, A, n, b, x, eps, max_iters);
    else
        slave_mpi_task(lin_sys, displs);

    return st;
}