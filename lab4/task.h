#pragma once

#include <cmath>
#include <vector>
#include <mpi.h>

namespace app
{
    class Task
    {
    private:
        int rank, size;
        int n;
        int layersLocal;
        int layerSize;
        int startZ;
        double h;

        const std::vector<int> sendCounts;
        const std::vector<int> displs;

        std::vector<double> matrix;
        std::vector<double> buff;

        double maxDelta;

    public:
        Task(int n, int r, int s, std::vector<int> sc, std::vector<int> ds);
        bool Run(double a);

    private:
        void runIteration(double a, double h, double mainCoeff);
        void calcWithin(double a, double h, double mainCoeff);
        void calcEdges(double a, double h, double mainCoeff);
        bool checkExit();
        void resetEps() { this->maxDelta = 0; }

        void calcCell(double x, double y, double z, double lz, double h, double mainCoeff);
        double calcPhi(int idx, double coeff, double h, double rho);
        double calcRho(double x, double y, double z, double h) { return std::sqrt(x * x + y * y + z * z) * h; };

        std::vector<MPI_Request> shareEdges();
        void waitEdges(std::vector<MPI_Request> requests) {MPI_Waitall(4, requests.data(), MPI_STATUSES_IGNORE);};
    };

} // namespace app