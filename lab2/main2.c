#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "utils/io_utils.h"
#include "utils/matrix.h"
#include "config.h"

void solve_variant2(TLinearSystem *sys, double *x)
{
    int n = sys->n;
    double *A = sys->A;
    double *b = sys->b;
    double *diffs = (double *)malloc(n * sizeof(double));

    double b_norm = 0.0;
    
    #pragma omp parallel for reduction(+:b_norm)
    for (int i = 0; i < n; i++)
    {
        b_norm += b[i] * b[i];
    }
    b_norm = sqrt(b_norm);

    int stop = 0;
    double global_norm = 0.0;

    #pragma omp parallel
    {
        int iter = 0;
        while (!stop && iter < MAX_ITERATIONS)
        {
            #pragma omp single
            global_norm = 0.0;

            #pragma omp for reduction(+:global_norm)
            for (int i = 0; i < n; i++)
            {
                double sum = -b[i];
                for (int j = 0; j < n; j++)
                {
                    sum += A[i * n + j] * x[j];
                }
                diffs[i] = sum;
                global_norm += sum * sum;
            }

            #pragma omp single
            {
                if (sqrt(global_norm) / b_norm < EPS)
                {
                    stop = 1;
                }
                iter++;
            }

            if (stop)
            {
                break;
            }

            #pragma omp for
            for (int i = 0; i < n; i++)
            {
                x[i] -= TAU * diffs[i];
            }
        }
    }
    free(diffs);
}

int main(int argc, char **argv)
{
    const char *filename = "matrix.txt";
    int n = get_lin_system_size(filename);
    if (n <= 0)
        return 1;

    TLoadRange range = {0, n, 0, n};
    TLinearSystem sys = read_lin_system(filename, range);
    if (!sys.A)
        return 1;

    double *x = (double *)calloc(n, sizeof(double));

    double start = omp_get_wtime();
    solve_variant2(&sys, x);
    double end = omp_get_wtime();

    printf("%f\n", end - start);

    free(x);
    free_lin_system(&sys);
    return 0;
}