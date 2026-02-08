#include "utils/i_reader.h"

#include <stdio.h>
#include <stdlib.h>

LinearSystem get_linear_system(char *path)
{
    LinearSystem sys = read_lin_system(path);
    return sys;
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
    if (A == NULL || b == NULL)
    {
        printf("Failed to read system\n");
        return 1;
    }

    printf("Matrix A:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%8.4lf ", A[i * n + j]);
        }
        printf("\n");
    }

    printf("\nVector b:\n");
    for (int i = 0; i < n; i++)
    {
        printf("%8.4lf\n", b[i]);
    }

    return 0;
}
