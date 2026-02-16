add_library(solver STATIC lin_solver.c)

target_link_libraries(solver PRIVATE matrix)