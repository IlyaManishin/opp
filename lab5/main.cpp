#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <thread>
#include <vector>

constexpr int TASK_COUNT = 500;
constexpr int ITERATION_COUNT = 10;

constexpr int TAG_NO_TASKS = -1;
constexpr int TAG_WORK_REQ = 1;
constexpr int TAG_WORK_SIZE = 2;
constexpr int TAG_WORK_DATA = 3;

constexpr bool IS_BALANCING_ENABLED = true;

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

std::thread runMessageThread(SafeQueue &queue, int size, int rank)
{
    return std::thread([&queue, size, rank]()
                       {
        while (queue.isRunning()) {
            int flag = 0;
            MPI_Status status;
            
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORK_REQ, MPI_COMM_WORLD, &flag, &status);

            if (flag) {
                int recvRank;
                MPI_Recv(&recvRank, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int available = queue.getSize();
                int give = available > 1 ? available / 2 : 0;

                std::vector<int> taskList;
                if (give > 0) {
                    taskList.resize(give);
                    int count = 0;
                    while (count < give && queue.tryPop(taskList[count])) {
                        count++;
                    }
                    give = count; 
                }

                int taskCount = (give == 0) ? TAG_NO_TASKS : give;
                MPI_Send(&taskCount, 1, MPI_INT, recvRank, TAG_WORK_SIZE, MPI_COMM_WORLD);

                if (taskCount > 0) {
                    MPI_Send(taskList.data(), taskCount, MPI_INT, recvRank, TAG_WORK_DATA, MPI_COMM_WORLD);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } });
}

std::thread runExecutingThread(SafeQueue &queue, int size, int rank)
{
    return std::thread([&queue, size, rank]()
                       {
        for (int i = 0; i < ITERATION_COUNT; ++i) {
            if (rank == 0)
                initTasks(queue, size, rank, i);

            std::cout << "PR[" << rank << "]: init = " << queue.getSize() << " tasks" << std::endl;

            executeTasks(queue);

            if (size != 1) {
                for (int j = 0; j < size; ++j) {
                    if (j == rank) {
                        continue;
                    }

                    MPI_Send(&rank, 1, MPI_INT, j, TAG_WORK_REQ, MPI_COMM_WORLD);

                    int taskGiven = 0;
                    MPI_Recv(&taskGiven, 1, MPI_INT, j, TAG_WORK_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (taskGiven != TAG_NO_TASKS && taskGiven > 0) {
                        std::vector<int> receivedTasks(taskGiven);
                        MPI_Recv(receivedTasks.data(), taskGiven, MPI_INT, j, TAG_WORK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        std::cout << "PR[" << rank << "]: process " << rank << " got " << taskGiven << " tasks from proc " << j << std::endl;

                        for (int val : receivedTasks) {
                            queue.push(val);
                        }

                        executeTasks(queue);
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::cout << "eT[" << rank << "]: all iterations done." << std::endl;
        queue.setRunning(false); });
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "size: " << size << ", tasks for each process: " << TASK_COUNT << ", iterations: " << ITERATION_COUNT << std::endl;
    }

    SafeQueue queue;
    queue.setRunning(true);

    double start = MPI_Wtime();

    std::thread messageThread = runMessageThread(queue, size, rank);
    std::thread executingThread = runExecutingThread(queue, size, rank);

    executingThread.join();
    messageThread.join();

    double finish = MPI_Wtime();
    double time = finish - start;
    double maxTime = 0;

    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "total time: " << maxTime << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}