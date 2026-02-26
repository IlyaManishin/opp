#include "../solver_math.h"

#include "matrix.h"
#include "mpi.h"

void mat_vec_task(
    int rank,
    int size,
    int *displs,
    int n,

    double *mat_part,
    double *v_part,
    double *d_part)
{
    for (int i = 0; i < displs[rank]; i++)
        d_part[i] = 0;

    int befRank = (rank - 1 + size) % size;
    int nextRank = (rank + 1) % size;

    int curPart = rank;
    int befPart = (rank - 1 + size) % size;
    for (int i = 0; i < size; i++)
    {
        int localCount = displs[curPart];
        int befCount = displs[befPart];

        matrix_mul_vec(mat_part, n, n, localCount, v_part, d_part);

        MPI_Send(v_part, localCount, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD);
        MPI_Send(d_part, localCount, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD);

        MPI_Recv(v_part, befCount, MPI_DOUBLE, befRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(d_part, befCount, MPI_DOUBLE, befRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        curPart = befPart;
        befPart = (rank - i - 1 + size) % size;
    }
}
