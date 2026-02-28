add_library(matrix STATIC matrix.c)
target_link_libraries(matrix PRIVATE m)
