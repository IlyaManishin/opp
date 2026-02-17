#include "lin_solver.h"
#include "../matrix/matrix.h"
#include "../utils/io_utils.h"
#include "solver_math.h"

#include <cblas.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

enum SLAVE_COMMANDS
{
    SLAVE_EXIT,
    SLAVE_MUL
};

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
        matrix_mul_vec(A, n, n, x, y);
        for (int i = 0; i < n; i++)
            y[i] -= b[i];

        double norm_r = vec_norm(y, n);
        if (norm_r / norm_b < eps)
        {
            vector_free(y);
            vector_free(Ay);
            return SOL_OK;
        }

        matrix_mul_vec(A, n, n, y, Ay);

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

void slave_task(TLinearSystem lin_sys, int *displs)
{
    int local_count = lin_sys.A_rows_count;
    double *A_part = lin_sys.A;
    double *b = lin_sys.b;
    int n = lin_sys.n;

    double *vec = malloc(sizeof(double) * n);
    double *out_local = malloc(sizeof(double) * local_count);

    if (!vec || !out_local)
    {
        free(vec);
        free(out_local);
        return;
    }

    while (1)
    {
        int cmd;
        MPI_Recv(&cmd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (cmd == SLAVE_EXIT)
            break;

        if (cmd == SLAVE_MUL)
        {
            MPI_Recv(vec, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            matrix_mul_vec(A_part, local_count, n, vec, out_local);

            MPI_Send(out_local, local_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    free(vec);
    free(out_local);
}

static void exit_slaves(int rank, int size)
{
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0)
        return;
    int cmd = SLAVE_EXIT;
    for (int dest = 1; dest < size; ++dest)
        MPI_Send(&cmd, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}

static void distributed_matvec(
    const double *A,
    const int *slaves_mask,
    int size,
    const int *displs,
    int n,
    double *x,
    double *dest,
    double *out_master) // remove this
{
    int cmd = SLAVE_MUL;
    for (int sl = 1; sl < size; ++sl)
    {
        MPI_Send(&cmd, 1, MPI_INT, sl, 0, MPI_COMM_WORLD);
        MPI_Send(x, n, MPI_DOUBLE, sl, 0, MPI_COMM_WORLD);
    }

    if (slaves_mask[0] > 0)
        matrix_mul_vec(A, slaves_mask[0], n, x, out_master);

    for (int i = 0; i < slaves_mask[0]; ++i)
        dest[i] = out_master[i];

    for (int sl = 1; sl < size; ++sl)
        MPI_Recv(dest + displs[sl], slaves_mask[sl], MPI_DOUBLE, sl, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

SolverStatus solve_linear_multy_impl(
    const int *slaves_mask,
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters)
{
    SolverStatus status = SOL_INVALID;

    if (!check_params(A, n, b, x, eps, max_iters))
        return SOL_INPUT_ERR;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 0 || rank != 0)
        return SOL_INVALID;

    if (size == 1)
    {
        return solve_linear_single_impl(A, n, b, x, eps, max_iters);
    }

    int *displs = malloc(sizeof(int) * size);
    double *y = malloc(sizeof(double) * n);
    double *Ay = malloc(sizeof(double) * n);
    double *out_master = malloc(sizeof(double) * slaves_mask[0]);

    if (!displs || !y || !Ay || !out_master)
        goto cleanup;

    displs[0] = 0;
    for (int i = 1; i < size; ++i)
        displs[i] = displs[i - 1] + slaves_mask[i - 1];

    for (int iter = 0; iter < max_iters; ++iter)
    {
        double norm_b = vec_norm(b, n);
        if (norm_b < 1e-30)
            norm_b = 1.0;

        distributed_matvec(A, slaves_mask, size, displs, n, x, y, out_master);

        for (int i = 0; i < n; ++i)
            y[i] -= b[i];

        double norm_r = vec_norm(y, n);
        if (norm_r / norm_b < eps)
        {
            status = SOL_OK;
            break;
        }

        distributed_matvec(A, slaves_mask, size, displs, n, y, Ay, out_master);

        double num = vec_dot(y, Ay, n);
        double den = vec_dot(Ay, Ay, n);
        if (fabs(den) < 1e-30)
        {
            status = SOL_INVALID;
            break;
        }

        double tau = num / den;
        for (int i = 0; i < n; ++i)
            x[i] -= tau * y[i];
    }

cleanup:
    exit_slaves(rank, size);

    free(displs);
    free(y);
    free(Ay);
    free(out_master);

    return status;
}
