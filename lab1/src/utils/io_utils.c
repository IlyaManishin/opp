#include "io_utils.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

void free_lin_system(TLinearSystem *sys)
{
    if (!sys)
        return;
    matrix_free(sys->A);
    vector_free(sys->b);
    vector_free(sys->r);

    sys->A = NULL;
    sys->b = NULL;
    sys->r = NULL;
    sys->n = 0;
}

int get_lin_system_size(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return -1;

    int size = -1;
    fscanf(f, "%d", &size);

    fclose(f);
    return size;
}

static bool read_matrix_part(FILE *f, double *A, int n, TLoadRange range)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double val = 0.0;
            if (fscanf(f, "%lf", &val) != 1)
                return false;

            if (i >= range.A_StartRow && i < range.A_EndRow)
                A[(i - range.A_StartRow) * n + j] = val;
        }
    }
    return true;
}

static bool read_vector_part(FILE *f, double *v, int n, int start, int end)
{
    for (int i = 0; i < n; i++)
    {
        double val = 0.0;
        if (fscanf(f, "%lf", &val) != 1)
            return false;

        if (i >= start && i < end)
            v[i - start] = val;
    }
    return true;
}

TLinearSystem read_lin_system(const char *filename, TLoadRange range)
{
    TLinearSystem sys = {0, NULL, NULL, NULL};

    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("fopen");
        return sys;
    }

    int n = 0;
    if (fscanf(f, "%d", &n) != 1 || n <= 0)
    {
        printf("Failed to read matrix size\n");
        goto error;
    }

    if (range.A_StartRow < 0 || range.A_EndRow > n || range.A_StartRow > range.A_EndRow ||
        range.b_Start < 0 || range.b_End > n || range.b_Start > range.b_End)
    {
        printf("Invalid load range\n");
        goto error;
    }

    int local_A = range.A_EndRow - range.A_StartRow;
    int local_b = range.b_End - range.b_Start;

    sys.n = n;

    sys.A = matrix_create(local_A, n);
    sys.b = vector_create(local_b);
    sys.r = vector_create(local_b);

    if (sys.A == NULL || sys.b == NULL || sys.r == NULL)
    {
        printf("Memory allocation failed\n");
        goto error;
    }

    if (!read_matrix_part(f, sys.A, n, range))
    {
        printf("Failed to read matrix A\n");
        goto error;
    }

    if (!read_vector_part(f, sys.b, n, range.b_Start, range.b_End))
    {
        printf("Failed to read vector b\n");
        goto error;
    }

    if (!read_vector_part(f, sys.r, n, range.b_Start, range.b_End))
    {
        printf("Failed to read vector r\n");
        goto error;
    }

#ifdef DEBUG
    if (sys.A && sys.b)
    {
        printf("Matrix A:\n");
        for (int i = 0; i < local_A; i++)
        {
            for (int j = 0; j < sys.n; j++)
            {
                printf("%8.4lf ", sys.A[i * sys.n + j]);
            }
            printf("\n");
        }

        printf("\nVector b:\n");
        for (int i = 0; i < local_b; i++)
        {
            printf("%8.4lf\n", sys.b[i]);
        }
        printf("\n");
    }
#endif

    fclose(f);
    return sys;
error:
    free_lin_system(&sys);
    fclose(f);
    return (TLinearSystem){0, NULL, NULL, NULL};
}

void writeAnswer(const char *destPath, double *x, int n)
{
    FILE *f = fopen(destPath, "w");
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
