cmake_minimum_required(VERSION 3.0.2)
project(dbscan)

find_package(catkin REQUIRED)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

add_executable(angle_test src/examples/angle_test.cpp)
add_executable(position_test src/examples/position_test.cpp)
