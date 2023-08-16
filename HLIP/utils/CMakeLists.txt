cmake_minimum_required(VERSION 3.10)

# Project name
project(robot_utils)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)  # Need c++ 17 for inline variables
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

# Add necessary packages
find_package(Eigen3 REQUIRED)

# Include the header files
include_directories(include)

# Create the mujoco interface library
add_library(robot_utils SHARED
    src/utility_kinematics.cpp
    src/utility_math.cpp
    src/utility_log.cpp
)

# Link the libraries to the mujoco_interface library
target_link_libraries(robot_utils PUBLIC Eigen3::Eigen)