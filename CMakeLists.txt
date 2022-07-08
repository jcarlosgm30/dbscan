cmake_minimum_required(VERSION 3.0.2)
project(dbscan)

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE nanoflann cxx_std_11)


target_include_directories(dbscan
	INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)

install(TARGETS dbscan
        EXPORT dbscanTargets)

add_library(dbscan::dbscan ALIAS dbscan)