#include "../matrix/matrix.h"
#include "../utils/io_utils.h"
#include "solver.h"
#include "solver_math.h"
#include "mpi_solver.h"

#include <math.h>
#include <mpi.h>
#include <stdlib.h>

SolverStatus solve_linear_single_impl(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters)
{
    if (!check_params(A, n, b, x, eps, max_iters))
        return SOL_INPUT_ERR;

    double *y = vector_create(n);
    double *Ay = vector_create(n);

    if (!y || !Ay)
    {
        free(y);
        free(Ay);
        return SOL_INVALID;
    }

    double norm_b = vec_norm(b, n);
    if (norm_b < 1e-30)
        norm_b = 1.0;

    for (int iter = 0; iter < max_iters; iter++)
    {
        matrix_mul_vec(A, n, n, n, x, y);
        for (int i = 0; i < n; i++)
            y[i] -= b[i];

        double norm_r = vec_norm(y, n);
        if (norm_r / norm_b < eps)
        {
            vector_free(y);
            vector_free(Ay);
            return SOL_OK;
        }

        matrix_mul_vec(A, n, n, n, y, Ay);

        double den = vec_dot(Ay, Ay, n);
        double num = vec_dot(y, Ay, n);

        if (fabs(num) < 1e-30)
        {
            vector_free(y);
            vector_free(Ay);
            return SOL_INVALID;
        }

        double tau = num / den;

        for (int i = 0; i < n; i++)
            x[i] -= tau * y[i];
    }

    vector_free(y);
    vector_free(Ay);
    return SOL_MAX_ITERS;
}

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
    {
        return solve_linear_single_impl(A, n, b, x, eps, max_iters);
    }

    if (rank == 0)
    {
        st = master_mpi_task(displs, A, n, b, x, eps, max_iters);
    }
    else
    {
        slave_mpi_task(lin_sys, displs);
    }
    return st;
}