cmake_minimum_required(VERSION 3.22)
project(HPC_MLP)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(HPC_MLP
        main.cpp
        MLPMath.cpp
        MLPMath.h
        MLPNet.cpp
	p4_f32vec4.h
	vec_arithmetic.h
	std_alloc.h
        MLPNet.h)
