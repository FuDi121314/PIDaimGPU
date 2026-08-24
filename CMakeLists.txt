cmake_minimum_required(VERSION 3.10)
project(AutoAim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/opt/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(AutoAim main.cpp dnn_detector.h dnn_detector.cpp)
target_sources(AutoAim PRIVATE resource.rc) 
target_link_libraries(AutoAim ${OpenCV_LIBS} user32 gdi32 ws2_32 winmm Shcore)