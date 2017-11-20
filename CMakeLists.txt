cmake_minimum_required(VERSION 3.6)
project(liblda)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-O4 -fopenmp -msse2 -msse3 -msse4")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O4 -fopenmp -msse2 -msse3 -msse4")

include_directories(${PROJECT_SOURCE_DIR}/src/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

file(GLOB SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*/*.cpp")
file(GLOB INCLUDE_FILES "${PROJECT_SOURCE_DIR}/src/include/*/*.h")

set(SRCS ${SOURCE_FILES} ${INCLUDE_FILES} src/include/corpus/corpus.h)

add_executable(liblda ${SRCS})
