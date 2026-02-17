add_library(solver STATIC single_solver.c multy_solver.c solver_math.c)

target_link_libraries(solver PRIVATE matrix)