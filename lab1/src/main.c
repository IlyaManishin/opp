#include "config.h"
#include "matrix/lin_solver.h"
#include "utils/i_reader.h"
#include "utils/logger.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

const char *RES_PATH = "output.txt";

LinearSystem get_linear_system(char *path)
{
    LinearSystem sys = read_lin_system(path);

#ifdef DEBUG
    if (sys.A && sys.b)
    {
        printf("Matrix A:\n");
        for (int i = 0; i < sys.n; i++)
        {
            for (int j = 0; j < sys.n; j++)
            {
                printf("%8.4lf ", sys.A[i * sys.n + j]);
            }
            printf("\n");
        }

        printf("\nVector b:\n");
        for (int i = 0; i < sys.n; i++)
        {
            printf("%8.4lf\n", sys.b[i]);
        }
        printf("\n");
    }
#endif
    return sys;
}

bool checkAnswer(double *check, double *valid, int n)
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

void writeAnswer(double *x, int n)
{
    FILE *f = fopen(RES_PATH, "w");
    if (!f)
    {
        perror("fopen");
        return;
    }
    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++)
    {
        fprintf(f, "%0.8lf\n", x[i]);
    }
    fclose(f);
}

int *get_rows_mask(int n, int size)
{
    int *counts = NULL;

    counts = malloc(size * sizeof(int));

    int base = n / size;
    int rem = n % size;

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        counts[i] = base + (i < rem ? 1 : 0);
    }
    return counts;
}

int get_slave_row_count(int n, int rank, int size)
{
    int base = n / size;
    int rem = n % size;

    int rows_count = base + (rank < rem ? 1 : 0);
    return rows_count;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    int statusCode = EXIT_SUCCESS;
    LinearSystem data;
    LOG_TIME(data = get_linear_system(argv[1]);)
    
    int n = data.n;
    double *A = data.A;
    double *b = data.b;
    double *x = (double *)malloc(n * sizeof(double));

    SolverStatus st;
    bool isMaster = true;
#ifdef MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    log_init(rank);

    int *rows_mask = get_rows_mask(n, size);
    if (rank == 0)
    {
        printf("Use MPI\n");
        st = solve_linear_multy(rows_mask, A, n, b, x, EPS, MAX_ITER);
    }
    else
    {
        isMaster = false;

        int row_offset = 0;
        for (int i = 0; i < rank; i++)
        {
            row_offset += rows_mask[i];
        }
        double *A_part = A + n * row_offset;
        int slave_rows = get_slave_row_count(n, rank, size);
        slave_task(A_part, n, slave_rows);
    }
    free(rows_mask);

    MPI_Finalize();
#else
    printf("No use MPI\n");
    st = solve_linear_single(A, n, b, x, EPS, MAX_ITER);
#endif

    if (isMaster)
    {
        if (st != SOL_OK)
        {
            printf("\nERROR - %d\n", (int)st);
            statusCode = EXIT_FAILURE;
        }
        else
        {
            statusCode = checkAnswer(x, b, n) ? EXIT_SUCCESS : EXIT_FAILURE;
            writeAnswer(x, n);
        }
    }

    free(A);
    free(b);
    free(x);
    return statusCode;
}
