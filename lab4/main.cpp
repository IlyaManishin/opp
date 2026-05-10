#include "config.h"
#include "task.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#define A_COEFF 10

using namespace app;

void get_displs(int n, int size, std::vector<int> &sendCounts, std::vector<int> &displs)
{
    sendCounts.resize(size);
    displs.resize(size);

    int layerSize = n * n;
    int rawCount = n / size;
    int remainder = n % size;

    int current_displacement = 0;

    for (int i = 0; i < size; i++)
    {
        int layersCount = rawCount + (i < remainder ? 1 : 0);

        sendCounts[i] = layersCount * layerSize;
        displs[i] = current_displacement;

        current_displacement += sendCounts[i];
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sendCounts;
    std::vector<int> displs;

    get_displs(N, size, sendCounts, displs);

    Task task(N, rank, size, sendCounts, displs);
    bool resp = task.Run(A_COEFF);

    int status;
    if (!resp)
    {
        std::cout << "Finish successfully" << std::endl;
        status = EXIT_SUCCESS;
    }
    else
    {

        std::cout << "Finish with errors" << std::endl;
        status = EXIT_FAILURE;
    }
    MPI_Finalize();
    
    return status;
}
