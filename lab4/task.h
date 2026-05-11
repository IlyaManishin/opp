#pragma once

#include <cmath>
#include <mpi.h>
#include <vector>

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
        const std::vector<double> &GetMatrix() { return this->matrix; };

    private:
        void runIteration(double mainCoef, double h, double a);
        void calcWithin(double mainCoef, double h, double a);
        void calcEdges(double mainCoef, double h, double a);
        bool checkAllExit();
        void resetEps() { this->maxDelta = 0; }

        void calcCell(double x, double y, double z, double lz, double h, double mainCoeff, double a);
        double calcPhi(int idx, double coeff, double h, double rho);
        double calcRho(double x, double y, double z, double a, double h) { return 6 - (x * x + y * y + z * z) * a * h * h; };

        std::vector<MPI_Request> shareEdges();
        void waitEdges(std::vector<MPI_Request> requests) { MPI_Waitall(4, requests.data(), MPI_STATUSES_IGNORE); };
    };

} // namespace app