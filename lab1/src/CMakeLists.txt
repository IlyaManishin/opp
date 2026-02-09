add_executable(lab1
    main.c
    matrix/matrix.c
    matrix/solver.c
    utils/i_reader.c
)
configure_file(gen/data.txt
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data.txt
               COPYONLY)

target_link_libraries(lab1 PRIVATE BLAS::BLAS m)
target_link_libraries(lab1 PRIVATE MPI::MPI_C)
