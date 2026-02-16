add_subdirectory(matrix)
add_subdirectory(utils) 
add_subdirectory(solver)

configure_file(gen/data.txt
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data.txt
               COPYONLY)

find_package(MPI REQUIRED)

#---------SINGLE CORE VERSION---------
add_executable(lab1_single main.c)
target_link_libraries(lab1_single PRIVATE matrix utils solver)
target_link_libraries(lab1_single PRIVATE BLAS::BLAS MPI::MPI_C m)
target_compile_definitions(lab1_single PRIVATE MPI)

#---------MULTY CORE VERSION---------
add_executable(lab1_multy main.c)
target_link_libraries(lab1_multy PRIVATE matrix utils solver)
target_link_libraries(lab1_multy PRIVATE BLAS::BLAS MPI::MPI_C m)
target_compile_definitions(lab1_multy PRIVATE MPI)