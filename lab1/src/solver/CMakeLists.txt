#-------SOLVERS BASE--------

add_library(solver_base OBJECT
    solver_single.c
    utils.c
)
target_link_libraries(solver_base PRIVATE matrix)
add_library(solver_mpi_base STATIC $<TARGET_OBJECTS:solver_base> solver_mpi.c)

#-------SOLVERS--------
add_library(solver_single STATIC $<TARGET_OBJECTS:solver_base>)

add_library(solver_mpi_comm STATIC $<TARGET_OBJECTS:solver_base> solver_mpi.c mpi_solvers/mpi_solver_comm.c)
target_link_libraries(solver_mpi_comm PUBLIC solver_mpi_base)

# add_library(solver_split STATIC mpi_solvers/mpi_solver_split.c mpi_solvers/split_tasks.c)
# target_link_libraries(solver_split PUBLIC solver_base)

