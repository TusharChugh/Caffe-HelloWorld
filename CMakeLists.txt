cmake_minimum_required(VERSION 2.8)
project(Caffe_HelloWorld)

set(CMAKE_CXX_STANDARD 11)
option(CAFFE_LIMITED_INSTALL "Only install built library" ON)
add_subdirectory(external/caffe/)
set(SOURCE_FILES main.cpp)

add_executable(Caffe_HelloWorld ${SOURCE_FILES})
# Find and link Caffe
if (TARGET caffe)
    target_link_libraries(${PROJECT_NAME} caffe)
else()
    find_package(Caffe REQUIRED)
    include_directories(${Caffe_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
endif()


