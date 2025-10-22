cmake_minimum_required(VERSION 3.10)
project(CoMN)

set(CMAKE_CXX_STANDARD 17)

# 编译当前目录下的所有.cpp文件
file(GLOB SOURCES *.cpp)

# 添加编译源文件
add_executable(main ${SOURCES})

# 将 CoMN_Sim 源文件添加到可执行文件中
target_sources(main PRIVATE ${GLOB})