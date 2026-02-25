#include "../matrix/matrix.h"
#include "../utils/io_utils.h"
#include "solver.h"
#include "solver_math.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum SLAVE_COMMANDS
{
    SLAVE_EXIT,
    SLAVE_MUL
};

void slave_mpi_task(TLinearSystem lin_sys, int *displs)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A_part = lin_sys.A;
    double *b = lin_sys.b;
    int n = lin_sys.n;
    int local_count = rank + 1 < size ? displs[rank + 1] - displs[rank] : n - displs[rank];

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

        switch (cmd)
        {
        case SLAVE_EXIT:
            goto exit;
        case SLAVE_MUL:
            MPI_Recv(vec, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matrix_mul_vec(A_part, local_count, n, vec, out_local);
            MPI_Send(out_local, local_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            break;
        default:
            goto exit;
        }
    }
    
exit:
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

static void mat_vec_task(
    const double *A,
    const int *displs,
    const int *slaves_mask,
    int size,
    int n,
    double *x,
    double *dest)
{
    int cmd = SLAVE_MUL;
    for (int sl = 1; sl < size; ++sl)
    {
        MPI_Send(&cmd, 1, MPI_INT, sl, 0, MPI_COMM_WORLD);
        MPI_Send(x, n, MPI_DOUBLE, sl, 0, MPI_COMM_WORLD);
    }

    matrix_mul_vec(A, slaves_mask[0], n, x, dest);

    for (int sl = 1; sl < size; ++sl)
        MPI_Recv(dest + displs[sl], slaves_mask[sl], MPI_DOUBLE, sl, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

SolverStatus master_mpi_task(
    const int *displs,
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



    double *y = malloc(sizeof(double) * n);
    double *Ay = malloc(sizeof(double) * n);
    int *slave_locals = (int *)malloc(sizeof(int) * size);

    if (slave_locals == NULL || y == NULL || Ay == NULL)
        goto cleanup;

    for (int i = 0; i < size - 1; i++)
    {
        slave_locals[i] = displs[i + 1] - displs[i];
    }
    slave_locals[size - 1] = n - displs[size - 1];

    for (int iter = 0; iter < max_iters; ++iter)
    {
        double norm_b = vec_norm(b, n);
        if (norm_b < 1e-30)
            norm_b = 1.0;

        mat_vec_task(A, displs, slave_locals, size, n, x, y);

        for (int i = 0; i < n; ++i)
            y[i] -= b[i];

        double norm_r = vec_norm(y, n);
        if (norm_r / norm_b < eps)
        {
            status = SOL_OK;
            break;
        }

        mat_vec_task(A, displs, slave_locals, size, n, y, Ay);

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

    free(slave_locals);
    free(y);
    free(Ay);

    return status;
}
