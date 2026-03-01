add_subdirectory(matrix)
add_subdirectory(utils) 
add_subdirectory(solver)

configure_file(gen/data.txt
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data.txt
               COPYONLY)

find_package(MPI REQUIRED)

#-------------------------------------
#---------SINGLE CORE VERSION---------

add_executable(sol_single main.c)
target_link_libraries(sol_single PRIVATE matrix utils solver_single)
target_link_libraries(sol_single PRIVATE m)


#------------------------------------
#---------MULTY CORE VERSION---------

#____COMMON_DATA____
add_executable(sol_multy_comm main.c)
target_link_libraries(sol_multy_comm PRIVATE matrix utils solver_mpi_comm)
target_link_libraries(sol_multy_comm PRIVATE MPI::MPI_C m)
target_compile_definitions(sol_multy_comm PRIVATE MPI)

#____SPLITTED_DATA____

add_executable(sol_multy_split main.c)
target_link_libraries(sol_multy_split PRIVATE matrix utils solver_mpi_split)
target_link_libraries(sol_multy_split PRIVATE MPI::MPI_C m)
target_compile_definitions(sol_multy_split PRIVATE MPI)

