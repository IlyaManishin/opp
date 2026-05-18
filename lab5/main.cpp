#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <thread>
#include <vector>

constexpr int TASK_COUNT = 3000;
constexpr int ITERATION_COUNT = 10;
constexpr int GIVE_MIN_LIMIT = 10;

constexpr int TASKS_EMPTY = -1;

constexpr int TAG_WORK_REQ = 1;
constexpr int TAG_WORK_SIZE = 2;
constexpr int TAG_WORK_DATA = 3;

static int isLog = true;

class SafeQueue
{
private:
    std::vector<int> m_storage;
    std::mutex m_mutex;
    bool m_isRunning = false;

public:
    void push(int task)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage.push_back(task);
    }

    bool tryPop(int &task)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_storage.empty())
        {
            return false;
        }
        task = m_storage.front();
        m_storage.erase(m_storage.begin());
        return true;
    }

    int getSize()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_storage.size();
    }

    bool isRunning()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_isRunning;
    }

    void setRunning(bool running)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_isRunning = running;
    }
};

void initTasks(SafeQueue &queue, int size, int rank, int iteration)
{
    int duration = (((iteration << 1) + 113) % 3) * 10;
    for (int i = 0; i < TASK_COUNT; ++i)
    {
        queue.push(duration);
    }
}

void executeTasks(SafeQueue &queue)
{
    int duration;
    while (queue.tryPop(duration))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(duration));
    }
}

void messageTask(SafeQueue &queue, int rank, int size)
{
    while (true)
    {
        int flag = 0;
        MPI_Status status;

        MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORK_REQ, MPI_COMM_WORLD, &flag, &status);

        if (flag)
        {
            int recvRank;
            MPI_Recv(&recvRank, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int available = queue.getSize();
            int give = available > GIVE_MIN_LIMIT ? available / 2 : 0;

            std::vector<int> taskList;
            int taskCount = 0;
            if (give > 0)
            {
                taskList.resize(give);
                while (taskCount < give && queue.tryPop(taskList[taskCount]))
                {
                    taskCount++;
                }
            }

            MPI_Send(&taskCount, 1, MPI_INT, recvRank, TAG_WORK_SIZE, MPI_COMM_WORLD);

            if (taskCount > 0)
            {
                MPI_Send(taskList.data(), taskCount, MPI_INT, recvRank, TAG_WORK_DATA, MPI_COMM_WORLD);
            }
        }
        else
        {
            if (!queue.isRunning())
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
}

bool tryGetTasks(SafeQueue &queue, int rank, int size)
{
    bool foundWork = false;

    for (int j = 0; j < size; ++j)
    {
        if (j == rank)
            continue;

        MPI_Send(&rank, 1, MPI_INT, j, TAG_WORK_REQ, MPI_COMM_WORLD);

        int taskGiven = 0;
        MPI_Recv(&taskGiven, 1, MPI_INT, j, TAG_WORK_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (taskGiven > 0)
        {
            std::vector<int> receivedTasks(taskGiven);
            MPI_Recv(receivedTasks.data(), taskGiven, MPI_INT, j, TAG_WORK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (isLog)
                std::cout << "PR[" << rank << "]: process " << rank << " got " << taskGiven << " tasks from proc " << j << std::endl;

            for (int val : receivedTasks)
            {
                queue.push(val);
            }

            foundWork = true;
            break;
        }
    }
    return foundWork;
}

void refillTask(SafeQueue &queue, int rank, int size)
{
    if (rank == 0)
        initTasks(queue, size, rank, 0);
    if (isLog)
        std::cout << "PR[" << rank << "]: init = " << queue.getSize() << " tasks" << std::endl;

    double start = MPI_Wtime();
    while (true)
    {
        if (queue.getSize() > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        bool isGot = tryGetTasks(queue, rank, size);
        if (!isGot)
        {
            while (queue.getSize() != 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            break;
        }
    }
    std::cout
        << "PR[" << rank << "]: iteration finished in "
        << (MPI_Wtime() - start)
        << " sec"
        << std::endl;

    MPI_Request barrier_req;
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
    int barrier_done = 0;
    while (!barrier_done)
    {
        MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
        if (!barrier_done)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    queue.setRunning(false);
}

void executingTask(SafeQueue &queue, int rank, int size)
{
    while (true)
    {
        while (queue.getSize() == 0 && queue.isRunning())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!queue.isRunning())
            break;

        executeTasks(queue);
    }
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc >= 2)
        isLog = std::atoi(argv[1]);

    if (rank == 0 && isLog)
    {
        std::cout << "size: " << size << ", tasks for each process: " << TASK_COUNT << ", iterations: " << ITERATION_COUNT << std::endl;
    }
    
    SafeQueue queue;
    queue.setRunning(true);

    double start = MPI_Wtime();

    std::thread messageThread = std::thread([&queue, size, rank]()
                                            { messageTask(queue, rank, size); });
    std::thread executingThread = std::thread([&queue, size, rank]()
                                              { executingTask(queue, rank, size); });
    std::thread refileThread = std::thread([&queue, size, rank]()
                                           { refillTask(queue, rank, size); });

    executingThread.join();
    messageThread.join();
    refileThread.join();

    double finish = MPI_Wtime();
    double time = finish - start;
    double maxTime = 0;

    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "PR[" << rank << "]: all iterations done." << std::endl;
        std::cout << "total time: " << maxTime << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}