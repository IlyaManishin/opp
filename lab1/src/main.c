#include "config.h"
#include "matrix/solver.h"
#include "utils/i_reader.h"

#include <assert.h>
#include <math.h>
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

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    LinearSystem data = get_linear_system(argv[1]);
    int n = data.n;
    double *A = data.A;
    double *b = data.b;
    double *x = (double *)malloc(n * sizeof(double));

#ifdef MPI
    SolverStatus st = solve_linear_multy(A, n, b, x, EPS, MAX_ITER);
#else
    SolverStatus st = solve_linear_single(A, n, b, x, EPS, MAX_ITER);
#endif

    int statusCode = EXIT_SUCCESS;
    if (st != SOL_OK)
    {
        printf("\nERROR\n");
        statusCode = EXIT_FAILURE;
    }
    else
    {
        statusCode = checkAnswer(x, b, n) ? EXIT_SUCCESS : EXIT_FAILURE;
        writeAnswer(x, n);
    }
    free(A);
    free(b);
    free(x);
    return statusCode;
}
