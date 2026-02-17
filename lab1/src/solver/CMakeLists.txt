add_library(solver STATIC lin_solver.c solver_utils.c)

target_link_libraries(solver PRIVATE matrix)