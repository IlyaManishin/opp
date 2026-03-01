#include <gtest/gtest.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

extern "C"
{
#include "../src/solver/mpi_solvers/split_tasks.h"
#include "../src/solver/utils.h"
}

TEST(MatVecParallel, AccuracyTest)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;
    int *displs = (int *)malloc(size * sizeof(int));
    int *counts = (int *)malloc(size * sizeof(int));

    int step = n / size;
    int rem = n % size;
    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        displs[i] = offset;
        counts[i] = step + (i < rem ? 1 : 0);
        offset += counts[i];
    }

    double *A_full = NULL;
    double *v_full = NULL;
    double *ref_res = NULL;

    if (rank == 0)
    {
        A_full = (double *)malloc(n * n * sizeof(double));
        v_full = (double *)malloc(n * sizeof(double));
        ref_res = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < n * n; i++)
            A_full[i] = (double)(i % 10);
        for (int i = 0; i < n; i++)
            v_full[i] = (double)(i % 5);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                ref_res[i] += A_full[i * n + j] * v_full[j];
    }

    int my_count = counts[rank];
    int max_count = 0;
    for (int i = 0; i < size; i++)
        if (counts[i] > max_count)
            max_count = counts[i];

    double *mat_part = (double *)malloc(my_count * n * sizeof(double));
    double *v_part = (double *)malloc(max_count * sizeof(double));
    double *d_part = (double *)malloc(my_count * sizeof(double));
    double *d_buf = (double *)malloc(my_count * sizeof(double));

    int *m_displs = (int *)malloc(size * sizeof(int));
    int *m_counts = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        m_displs[i] = displs[i] * n;
        m_counts[i] = counts[i] * n;
    }

    MPI_Scatterv(A_full, m_counts, m_displs, MPI_DOUBLE, mat_part, my_count * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(v_full, counts, displs, MPI_DOUBLE, v_part, my_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mat_vec_mul_task(rank, size, displs, max_count, n, mat_part, v_part, d_part, d_buf);

    double *final_res = NULL;
    if (rank == 0)
        final_res = (double *)malloc(n * sizeof(double));

    MPI_Gatherv(d_part, my_count, MPI_DOUBLE, final_res, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            EXPECT_NEAR(ref_res[i], final_res[i], 1e-9);
        }
        free(A_full);
        free(v_full);
        free(ref_res);
        free(final_res);
    }

    free(mat_part);
    free(v_part);
    free(d_part);
    free(d_buf);
    free(displs);
    free(counts);
    free(m_displs);
    free(m_counts);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}