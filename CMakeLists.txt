cmake_minimum_required(VERSION 3.24)
project(CP)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXE_LINKER_FLAGS "-ld_classic")

add_executable(CP "AHC045/main.tcpp")

# for stack size
#set(CMAKE_EXE_LINKER_FsLAGS "-Wl,-stack_size,0x20000000")