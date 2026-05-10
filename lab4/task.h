#pragma once

#include <vector>

namespace app
{
    class Task
    {
    private:
        int rank, size;
        int localCount;
        int layerCount;

        const std::vector<int> sendCounts;
        const std::vector<int> displs;

        std::vector<double> matrix;
        std::vector<double> buff;

    public:
        Task(int n, int r, int s, std::vector<int> sc, std::vector<int> ds);
        void Run();
    };

} // namespace app