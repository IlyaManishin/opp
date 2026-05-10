#include "task.h"
#include "config.h"

#include <mpi.h>

namespace app
{
    Task::Task(int n, int r, int s, std::vector<int> sc, std::vector<int> ds)
        : rank(r), size(s), sendCounts(sc), displs(ds)
    {
        this->localCount = sendCounts[rank];
        this->layerCount = n * n;

        matrix.resize(localCount + 2 * layerCount, DEFAULT_VALUE);
        buff.resize(localCount + 2 * layerCount, DEFAULT_VALUE);
    }

    void Task::Run()
    {
    }

} // namespace app
