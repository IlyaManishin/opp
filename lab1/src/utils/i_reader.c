#include "i_reader.h"
#include <stdio.h>
#include <stdlib.h>

static void free_system(LinearSystem *sys)
{
    if (!sys)
        return;
    free(sys->A);
    free(sys->b);
    sys->A = NULL;
    sys->b = NULL;
    sys->n = 0;
}

LinearSystem read_lin_system(const char *filename)
{
    LinearSystem sys = {0, NULL, NULL};

    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("fopen");
        return sys;
    }

    if (fscanf(f, "%d", &sys.n) != 1 || sys.n <= 0)
    {
        printf("Failed to read matrix size\n");
        fclose(f);
        return sys;
    }

    sys.A = (double *)malloc(sizeof(double) * sys.n * sys.n);
    sys.b = (double *)malloc(sizeof(double) * sys.n);
    if (sys.A == NULL || sys.b == NULL)
    {
        printf("Memory allocation failed\n");
        fclose(f);
        if (sys.A == NULL){
            free(sys.A);
            sys.A = NULL;
        }
        if (sys.A == NULL){
            free(sys.b);
            sys.b = NULL;
        }
        sys.n = 0;
        return sys;
    }

    for (int i = 0; i < sys.n; i++)
    {
        for (int j = 0; j < sys.n; j++)
        {
            if (fscanf(f, "%lf", &sys.A[i * sys.n + j]) != 1)
            {
                printf("Failed to read matrix element A[%d][%d]\n", i, j);
                free_system(&sys);
                fclose(f);
                return (LinearSystem){0, NULL, NULL};
            }
        }
    }

    for (int i = 0; i < sys.n; i++)
    {
        if (fscanf(f, "%lf", &sys.b[i]) != 1)
        {
            printf("Failed to read vector element b[%d]\n", i);
            free_system(&sys);
            fclose(f);
            return (LinearSystem){0, NULL, NULL};
        }
    }

    fclose(f);
    return sys;
}
