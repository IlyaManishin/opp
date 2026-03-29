cmake_minimum_required(VERSION 3.10)
project(OpenMP_Solver C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

configure_file(data.txt
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data.txt
    COPYONLY)
configure_file(runner.sh
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/runner.sh
    COPYONLY)

set(UTILS_SRC 
    utils/io_utils.c 
    utils/matrix.c
)

# --- build 1 ---
add_executable(parallel1 main1.c ${UTILS_SRC})
target_link_libraries(parallel1 PRIVATE OpenMP::OpenMP_C)
target_link_libraries(parallel1 PRIVATE m)

# --- build 2 ---
add_executable(parallel2 main2.c ${UTILS_SRC})
target_link_libraries(parallel2 PRIVATE OpenMP::OpenMP_C)
target_link_libraries(parallel2 PRIVATE m)

target_compile_options(parallel1 PRIVATE -O3)
target_compile_options(parallel2 PRIVATE -O3)