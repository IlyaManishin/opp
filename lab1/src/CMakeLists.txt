add_executable(lab1
    main.c
    matrix/matrix.c
    utils/i_reader.c
)
configure_file(input.txt
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/input.txt
               COPYONLY)

target_link_libraries(lab1 PRIVATE BLAS::BLAS)
