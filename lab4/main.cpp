#include "config.h"
#include "task.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

#define A_COEFF 10e5
#define SIZE_FILE "sizeN.txt"

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

bool check_result(const std::vector<double> &res, int localCount, int N)
{
    double maxDiff = 0;

    int testsCount = 5;
    int layerCount = N * N;
    int step = localCount / layerCount / testsCount;
    for (int i = 0; i < testsCount; i++)
    {
        int pos = 1 + step * i;

        int x = pos;
        int y = pos;
        int z = pos;

        int layerSize = N * N;
        int idx = z * layerSize + y * N + x;

        double h = 1.0 / N;
        double h2 = h * h;

        double phi = res[idx];
        double phi_x_plus = res[idx + 1];
        double phi_x_minus = res[idx - 1];
        double phi_y_plus = res[idx + N];
        double phi_y_minus = res[idx - N];
        double phi_z_plus = res[idx + layerSize];
        double phi_z_minus = res[idx - layerSize];

        double d2phi_dx2 = (phi_x_plus - 2.0 * phi + phi_x_minus) / h2;
        double d2phi_dy2 = (phi_y_plus - 2.0 * phi + phi_y_minus) / h2;
        double d2phi_dz2 = (phi_z_plus - 2.0 * phi + phi_z_minus) / h2;

        double left_side = d2phi_dx2 + d2phi_dy2 + d2phi_dz2 - (A_COEFF * phi);

        double real_x = x * h;
        double real_y = y * h;
        double real_z = z * h;

        double right_side = 6 - (real_x * real_x + real_y * real_y + real_z * real_z) * A_COEFF;
        double diff = std::abs(left_side - right_side);

        char positions[128];
        snprintf(positions, sizeof(positions), "(%d,%d,%d)", x, y, z);

        std::cout << "Verification at " << positions << ":" << std::endl;
        std::cout << "Left side (L(phi)): " << left_side << std::endl;
        std::cout << "Right side (rho):   " << right_side << std::endl;
        std::cout << "Absolute error:     " << diff << std::endl;

        maxDiff = std::max(diff, maxDiff);
    }

    return (maxDiff < 1e-3);
}

int get_grid_size()
{
    FILE *file = fopen(SIZE_FILE, "r");
    if (file == NULL)
        return -1;
    int N;
    fscanf(file, "%d", &N);
    fclose(file);
    return N;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = get_grid_size();
    if (N == -1)
    {
        std::cout << "Grid size file open error!" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> sendCounts;
    std::vector<int> displs;

    get_displs(N, size, sendCounts, displs);

    Task task(N, rank, size, sendCounts, displs);
    bool resp = task.Run(A_COEFF);
    if (rank == 0)
    {
        bool check = check_result(task.GetMatrix(), sendCounts[0], N);
        std::cout << "Test:" << (check ? "Passed" : "Failed") << std::endl;
    }

    int status;
    if (resp)
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
