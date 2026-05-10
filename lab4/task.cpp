    #include "task.h"
    #include "config.h"

    #include <mpi.h>
    #include <vector>
    #include <iostream>

    namespace app
    {
        Task::Task(int n, int r, int s, std::vector<int> sc, std::vector<int> ds)
            : rank(r), size(s), n(n), sendCounts(sc), displs(ds)
        {
            this->layerSize = n * n;
            this->layersLocal = sendCounts[rank] / layerSize;
            this->startZ = displs[rank] / layerSize;

            this->h = 1.0 / n;

            matrix.resize((layersLocal + 2) * layerSize, DEFAULT_VALUE);
            buff.resize((layersLocal + 2) * layerSize, DEFAULT_VALUE);
        }

        bool Task::Run(double a)
        {
            const double h2 = h * h;
            const double mainCoef = 1.0 / (6.0 + a * h2);

            for (int i = 0; i < MAX_ITERS; i++)
            {
                this->resetEps();

                runIteration(mainCoef, h);

                if (this->checkAllExit())
                    return true;
                    
                //  if (rank == 0) std::cout << std::to_string(this->maxDelta * 1000) << std::endl;
            }
            return false;
        }

        void Task::runIteration(double mainCoef, double h)
        {
            std::vector<MPI_Request> requests = this->shareEdges();

            this->calcWithin(mainCoef, h);
            this->waitEdges(requests);
            this->calcEdges(mainCoef, h);

            std::swap(matrix, buff);
        }

        void Task::calcWithin(double mainCoef, double h)
        {
            for (int lz = 2; lz < layersLocal; lz++)
            {
                int z = startZ + lz;
                if (z <= 0 || z >= n - 1)
                    continue;

                for (int y = 1; y < n - 1; y++)
                {
                    for (int x = 1; x < n - 1; x++)
                    {
                        calcCell(x, y, z, lz, h, mainCoef);
                    }
                }
            }
        }

        void Task::calcEdges(double mainCoef, double h)
        {
            int edgeLayers[2] = {1, layersLocal};
            int numEdges = (layersLocal > 1) ? 2 : 1;

            for (int i = 0; i < numEdges; i++)
            {
                int lz = edgeLayers[i];
                int z = startZ + lz;
                if (z <= 0 || z >= n - 1)
                    continue;

                for (int y = 1; y < n - 1; y++)
                {
                    for (int x = 1; x < n - 1; x++)
                    {
                        calcCell(x, y, z, lz, h, mainCoef);
                    }
                }
            }
        }

        bool Task::checkAllExit()
        {
            int localExit = (this->maxDelta < EXIT_DELTA);
            int globalExit;
            MPI_Allreduce(&localExit, &globalExit, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            return (bool)globalExit;
        }

        std::vector<MPI_Request> Task::shareEdges()
        {
            std::vector<MPI_Request> requests(4, MPI_REQUEST_NULL);

            if (rank > 0)
            {
                MPI_Irecv(&matrix[0], layerSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
                MPI_Isend(&matrix[layerSize], layerSize, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[1]);
            }

            if (rank < size - 1)
            {
                MPI_Irecv(&matrix[(layersLocal + 1) * layerSize], layerSize, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[2]);
                MPI_Isend(&matrix[layersLocal * layerSize], layerSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
            }
            return requests;
        }

        void Task::calcCell(double x, double y, double z, double lz, double h, double mainCoeff)
        {
            int idx = lz * layerSize + y * n + x;
            double rho = this->calcRho(x, y, z, h);

            double phi = matrix[idx];
            double phiNext = this->calcPhi(idx, mainCoeff, h, rho);
            double delta = std::abs(phiNext - phi);

            this->maxDelta = std::max(delta, this->maxDelta);
            this->buff[idx] = phiNext;
        }

        double Task::calcPhi(int idx, double coeff, double h, double rho)
        {
            double sum_neighbors =
                matrix[idx + 1] + matrix[idx - 1] +
                matrix[idx + n] + matrix[idx - n] +
                matrix[idx + layerSize] + matrix[idx - layerSize];

            return coeff * (sum_neighbors - h * h * rho);
        }

    } // namespace app
