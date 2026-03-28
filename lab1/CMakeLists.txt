cmake_minimum_required(VERSION 3.16)

project(lab1 C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_C_COMPILER mpecc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MPI REQUIRED)
include_directories(src/include /usr/local/include)
add_compile_options(-O2 -Wall -Wextra -Wpedantic)
link_directories(/usr/local/lib)
# add_definitions(-DDEBUG)

add_subdirectory(src/)
# add_subdirectory(tests/)
