#include "config.h"
#include "solver/solver.h"

#include "matrix.h"
#include "io_utils.h"
#include "logger.h"

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
    double max_err = -1;
    int max_i = -1;

    for (int i = 0; i < n; i++)
    {
        double err = fabs(check[i] - valid[i]);
        if (err > max_err)
        {
            max_err = err;
            max_i = i;
        }
    }

    if (max_err > EPS)
    {
        printf("Invalid: max error = %.10lf at i=%d: x[%d] = %.10lf, r[%d] = %.10lf\n",
               max_err, max_i, max_i, valid[max_i], max_i, check[max_i]);
        return false;
    }

    printf("Success\n");
    return true;
}


#ifdef MPI

static int *get_tasks_displs(int n, int size)
{
    int *displs = NULL;

    displs = malloc(size * sizeof(int));
    if (displs == NULL)
        return NULL;

    int base = n / size;
    int rem = n % size;

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

#endif

static bool solve_linear_system()
{
    bool succ = true;
    TLinearSystem linSys;
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
    log_init(rank);

    int *displs = get_tasks_displs(n, size);
    TLoadRange range = {.A_StartRow = displs[rank],
                        .A_EndRow = rank + 1 < size ? displs[rank + 1] : n,
                        .b_Start = 0,
                        .b_End = n};
    LOG_TIME(linSys = read_lin_system(SRC_PATH, range);)
    st = solve_mpi_impl(linSys, x, displs, EPS, MAX_ITER, rank, size);

    free(displs);

#else
    printf("No use MPI\n");

    isMaster = true;
    TLoadRange range = {.A_StartRow = 0,
                        .A_EndRow = n,
                        .b_Start = 0,
                        .b_End = n};
    LOG_TIME(linSys = read_lin_system(SRC_PATH, range);)
    st = solve_linear_single_impl(linSys, x, EPS, MAX_ITER);

#endif

    if (isMaster)
    {
        if (st != SOL_OK || !checkAnswer(x, linSys.r, linSys.n))
        {
            printf("\nERROR - %d\n", (int)st);
            succ = false;
        }
        writeAnswer(RES_PATH, x, n);
    }
    free_lin_system(&linSys);
    vector_free(x);
    return succ;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
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
