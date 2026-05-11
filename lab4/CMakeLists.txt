cmake_minimum_required(VERSION 3.10)

project(ParallelTask)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_compile_options("-Wall" "-Wextra" "-Wpedantic")

configure_file(sizeN.txt
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sizeN.txt
    COPYONLY)

add_executable(main 
    main.cpp 
    task.cpp
)

find_package(MPI REQUIRED)
target_link_libraries(main PRIVATE MPI::MPI_CXX)
