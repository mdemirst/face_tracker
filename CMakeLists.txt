cmake_minimum_required(VERSION 2.8)
project(face_tracker)
find_package(OpenCV 3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-narrowing")

MESSAGE(${OpenCV_VERSION})

add_executable(face_tracker src/face_tracker.cpp)
target_link_libraries(face_tracker ${OpenCV_LIBS})

