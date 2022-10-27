cmake_minimum_required(VERSION 3.0.0)
project(cpp_vcpkg_test VERSION 0.1.0)

#set(CMAKE_TOOLCHAIN_FILE=/home/melihozaydin/vcpkg/scripts/buildsystems/vcpkg.cmake)
include(/home/melihozaydin/vcpkg/scripts/buildsystems/vcpkg.cmake)

include(CTest)
enable_testing()

add_executable(cpp_vcpkg_test main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(OpenCV REQUIRED)
target_link_libraries(cpp_vcpkg_test ${OpenCV_LIBS})