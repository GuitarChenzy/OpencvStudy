cmake_minimum_required(VERSION 3.0.0)
project(MyCpp VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}") 

find_package(OpenCV REQUIRED)
# find_package(PCL 1.8 REQUIRED)

# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})

add_executable(MyCpp cv_tracing.cpp)

target_link_libraries(MyCpp ${OpenCV_LIBS})

