#include "task.h"
#include "config.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

using namespace app;

void get_displs(int n, int size, std::vector<int> &sendcounts, std::vector<int> &displs)
{
    sendcounts.resize(size);
    displs.resize(size);

    int plane_size = n * n;     
    int base_layers = n / size; 
    int remainder = n % size;  

    int current_displacement = 0;

    for (int i = 0; i < size; i++)
    {
        int layers_for_proc = base_layers + (i < remainder ? 1 : 0);

        sendcounts[i] = layers_for_proc * plane_size;
        displs[i] = current_displacement;

        current_displacement += sendcounts[i];
    }
}

int main(int argc, char const *argv[])
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sendCounts;
    std::vector<int> displs;

    get_displs(N, size, sendCounts, displs);
    
    Task task(N, rank, size, sendCounts, displs);
    task.Run();

    return 0;
}
