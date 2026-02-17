#include "config.h"

#include "matrix/matrix.h"
#include "solver/lin_solver.h"
#include "utils/io_utils.h"
#include "utils/logger.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

const char *SRC_PATH = "data.txt";
const char *RES_PATH = "output.txt";

static bool checkAnswer(const double *check, const double *valid, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(check[i] - valid[i]) > EPS)
        {
            printf("Invalid: %8.4lf, %8.4lf", valid[i], check[i]);
            return false;
        }
    }
    printf("Success\n");
    return true;
}

static int *get_tasks_displs(int n, int size)
{
    int *displs = NULL;

    displs = malloc(size * sizeof(int));
    if (displs == NULL)
        return NULL;

    int base = n / size;
    int rem = n % size;

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        displs[i] = base + (i < rem ? 1 : 0);
    }

    int accum = 0;
    for (int i = 0; i < size; i++)
    {
        int cur = accum;
        accum += displs[i];
        displs[i] = cur;
    }
    return displs;
}

static int get_slave_row_count(int n, int rank, int size)
{
    int base = n / size;
    int rem = n % size;

    int rows_count = base + (rank < rem ? 1 : 0);
    return rows_count;
}

static SolverStatus solve_mpi(TLinearSystem lin_sys, double *x, int *displs, int rank, int size)
{
    SolverStatus st = SOL_OK;

    double *A = lin_sys.A;
    double *b = lin_sys.b;
    int n = lin_sys.n;

    log_init(rank);

    if (rank == 0)
    {
        st = solve_linear_multy_impl(displs, A, n, b, x, EPS, MAX_ITER);
    }
    else
    {
        // slave_task(A, lin_sys.n, *displs);
    }
    return st;
}

static SolverStatus solve_single(TLinearSystem lin_sys, double *x)
{
    double *A = lin_sys.A;
    double *b = lin_sys.b;
    int n = lin_sys.n;

    SolverStatus st = solve_linear_single_impl(A, n, b, x, EPS, MAX_ITER);
    return st;
}

static bool solve_linear_system()
{
    bool succ = true;
    TLinearSystem lin_sys;
    int n = get_lin_system_size(SRC_PATH);
    double *x = vector_create(n);

    SolverStatus st;
    bool isMaster = false;
#ifdef MPI
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        isMaster = true;
        printf("Use MPI\n");
    }

    int *displs = get_tasks_displs(n, size);
    TLoadRange range = {.A_StartRow = displs[rank],
                        .A_EndRow = rank + 1 < size ? displs[rank + 1] : n,
                        .b_Start = 0,
                        .b_End = n};
    LOG_TIME(lin_sys = read_lin_system(SRC_PATH, range);)
    st = solve_mpi(lin_sys, x, displs, rank, size);

    free(displs);

#else
    isMaster = true;
    st = solve_single(lin_sys, x);
    printf("No use MPI\n");

#endif

    if (isMaster)
    {
        if (st != SOL_OK || !checkAnswer(x, lin_sys.r, lin_sys.n))
        {
            printf("\nERROR - %d\n", (int)st);
            succ = false;
        }
        writeAnswer(RES_PATH, x, n);
    }
    free_lin_system(&lin_sys);
    vector_free(x);
    return succ;
}

int main(int argc, char **argv)
{
#ifdef MPI
    MPI_Init(&argc, &argv);
#endif

    bool res = solve_linear_system();

#ifdef MPI
    MPI_Finalize();
#endif

    return res ? EXIT_SUCCESS : EXIT_FAILURE;
}
