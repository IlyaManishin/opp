#include "../utils.h"

#include "matrix.h"

#include <mpi.h>
#include <stdlib.h>

void mat_vec_task(
    int rank,
    int size,
    int *displs,
    int maxLocal,
    int n,

    double *mat_part,
    double *v_part,
    double *d_part,
    double *d_buf)
{
    int startLocal = LOCAL_COUNT(displs, rank, size, n);
    for (int i = 0; i < startLocal; i++)
        d_part[i] = 0;

    const int befRank = (rank - 1 + size) % size;
    const int nextRank = (rank + 1) % size;

    int befPart = befRank;
    int curPart = rank;
    for (int i = 0; i < size; i++)
    {
        int locCount = LOCAL_COUNT(displs, curPart, size, n);
        int befCount = LOCAL_COUNT(displs, befPart, size, n);

        matrix_mul_vec(mat_part + displs[curPart], startLocal, n, locCount, v_part, d_buf);
        vec_add(d_buf, d_part, d_part, startLocal);

        // MPI_Send(v_part, locCount, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD);
        // MPI_Recv(v_part, befCount, MPI_DOUBLE, befRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv_replace(
            v_part, 
            maxLocal,      
            MPI_DOUBLE, 
            nextRank, 0,    
            befRank, 0,     
            MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE
        );
        
        curPart = befPart;
        befPart = (befPart - 1 + size) % size;
    }
}
