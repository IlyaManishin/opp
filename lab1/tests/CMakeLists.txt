set(SRC_DIR ../src)
set(SOL_DIR ${SRC_DIR}/solver)


add_library(split_tasks_lib STATIC
    ${SOL_DIR}/mpi_solvers/split_tasks.c
    ${SRC_DIR}/matrix/matrix.c
)
add_executable(check tests.cpp)
target_link_libraries(check PUBLIC split_tasks_lib MPI::MPI_C m gtest)

