add_library(solver STATIC solver.c mpi_solver.c solver_math.c)
# add_library(solver_split STATIC solver.c mpi_solver.c solver_math.c)


target_link_libraries(solver PRIVATE matrix)