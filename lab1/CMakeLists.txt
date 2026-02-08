cmake_minimum_required(VERSION 3.16)

project(lab1 C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(BLAS REQUIRED)

# add_definitions(-DDEBUG)
add_definitions(-DMPI)

add_subdirectory(src/)


