#include "../solver.h"
#include "../utils.h"
#include "split_tasks.h"

#include "io_utils.h"
#include "matrix.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum SLAVE_COMMANDS
{
    SLAVE_EXIT,
    SLAVE_HANDLE_ITER,
    SLAVE_GATHER_X
} SLAVE_COMMANDS;

typedef enum IterStatus
{
    ITER_INVALID,
    ITER_OK,
} IterStatus;

typedef struct TTaskContext
{
    int rank;
    int size;
    int *displs;

    double *A_part;
    double *b_part;
    int n;
    double eps;
    int *localCounts; //???
    int localCount;
    int maxLocal;

    double *x_part;
    double *y_part;
    double *Ay_part;
    double *loc_buf;

    double norm_y;
} TTaskContext;

TTaskContext *get_task_context(TLinearSystem linSys, int *displs, double eps)
{
    TTaskContext *context = (TTaskContext *)malloc(sizeof(TTaskContext));
    if (context == NULL)
        return NULL;

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 0)
    {
        free(context);
        return NULL;
    }

    context->rank = rank;
    context->size = size;
    context->displs = displs;

    context->localCounts = (int *)malloc(size * sizeof(int));
    int localCount = LOCAL_COUNT(displs, rank, size, linSys.n);
    int maxPart = 0;
    for (int i = 0; i < size; i++)
    {
        int local = LOCAL_COUNT(displs, i, size, linSys.n);

        maxPart = max(local, maxPart);
        context->localCounts[i] = local;
    }
    context->maxLocal = maxPart;
    context->localCount = localCount;

    context->A_part = linSys.A;
    context->b_part = linSys.b + displs[rank];
    context->n = linSys.n;
    context->eps = eps;

    context->Ay_part = (double *)malloc(maxPart * sizeof(double));
    context->x_part = (double *)malloc(maxPart * sizeof(double));
    context->y_part = (double *)malloc(maxPart * sizeof(double));

    context->loc_buf = (double *)malloc(localCount * sizeof(double));

    if (context->Ay_part == NULL ||
        context->x_part == NULL ||
        context->y_part == NULL ||
        context->localCounts == NULL ||
        context->loc_buf == NULL)
    {
        free(context->Ay_part);
        free(context->x_part);
        free(context->y_part);
        free(context->localCounts);
        free(context->loc_buf);
        free(context);
        return NULL;
    }

    for (int i = 0; i < localCount; i++)
    {
        context->x_part[i] = 0;
    }

    return context;
}

void delete_context(TTaskContext *context)
{
    free(context->Ay_part);
    free(context->x_part);
    free(context->y_part);
    free(context->localCounts);
    free(context->loc_buf);

    free(context);
}

static void mpi_mat_vec_mul(TTaskContext *con, double *mat_part, double *v_part, double *d_part)
{
    mat_vec_mul_task(con->rank, con->size,
                     con->displs,
                     con->maxLocal,
                     con->n,
                     mat_part, v_part, d_part, con->loc_buf);
}

static void mpi_vec_vec_sub(TTaskContext *con, double *vec1, double *vec2, double *dest)
{
    vec_sub(vec1, vec2, dest, con->localCount);
}

static double mpi_vec_dot(TTaskContext *con, double *vec1, double *vec2)
{
    return vec_dot_task(vec1, vec2, con->localCount);
}

static void mpi_gather_result(TTaskContext *con, double *x)
{
    MPI_Gatherv(
        con->x_part,
        con->localCount,
        MPI_DOUBLE,
        x,
        con->localCounts,
        con->displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);
}

bool handle_iteration(TTaskContext *con)
{
    // Compute y = Ax - b
    mpi_mat_vec_mul(con, con->A_part, con->x_part, con->y_part);
    mpi_vec_vec_sub(con, con->y_part, con->b_part, con->y_part);

    // Compute y norm, check return condition
    double y_dot = mpi_vec_dot(con, con->y_part, con->y_part);
    con->norm_y = sqrt(y_dot);

    // Compute Ay
    mpi_mat_vec_mul(con, con->A_part, con->y_part, con->Ay_part);

    // Compute r = (y, Ay) / (Ay, Ay)
    double dot_y_Ay = mpi_vec_dot(con, con->Ay_part, con->y_part);
    double dot_Ay_Ay = mpi_vec_dot(con, con->Ay_part, con->Ay_part);

    if (fabs(dot_Ay_Ay) < 1e-30)
        return false;

    double tau = dot_y_Ay / dot_Ay_Ay;

    // compute x_part with (x - r*y)
    for (int i = 0; i < con->localCount; i++)
    {
        con->x_part[i] -= tau * con->y_part[i];
    }
    return true;
}

void slave_mpi_task(TLinearSystem lin_sys, int *displs, double eps)
{
    TTaskContext *con = get_task_context(lin_sys, displs, eps);
    if (con == NULL)
        return;

    while (1)
    {
        int cmd;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

        switch (cmd)
        {
        case SLAVE_HANDLE_ITER:
            handle_iteration(con);
            break;
        case SLAVE_GATHER_X:
            mpi_gather_result(con, NULL);
            break;
        case SLAVE_EXIT:
            goto exit;
        default:
            goto exit;
        }
    }

exit:
    delete_context(con);
}

static void exit_slaves()
{
    int cmd = SLAVE_EXIT;
    MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void master_gather_x(TTaskContext *con, double *x)
{
    int cmd = SLAVE_GATHER_X;
    MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_gather_result(con, x);
}

SolverStatus master_mpi_task(
    TLinearSystem linSys,
    double *x,
    int *displs,
    double eps,
    int maxIters)
{
    SolverStatus status = SOL_INVALID;
    const double *b = linSys.b;
    TTaskContext *con = get_task_context(linSys, displs, eps);

    double norm_b = vec_norm(b, con->n);
    if (norm_b < 1e-30)
        norm_b = 1.0;

    for (int iter = 0; iter < maxIters; ++iter)
    {
        int cmd = SLAVE_HANDLE_ITER;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

        bool isSucc = handle_iteration(con);
        if (!isSucc)
        {
            status = SOL_INVALID;
            goto exit;
        }
        if (con->norm_y / norm_b < eps)
        {
            status = SOL_OK;
            master_gather_x(con, x);
            goto exit;
        }
    }
    master_gather_x(con, x);
    status = SOL_MAX_ITERS;

exit:
    exit_slaves();
    delete_context(con);
    return status;
}
