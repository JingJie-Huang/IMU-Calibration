cmake_minimum_required( VERSION 2.8 )
project( IMU_Calibation )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

add_executable( Ceres_IMU_Calibration ceres_imu.cpp )
target_link_libraries( Ceres_IMU_Calibration ${CERES_LIBRARIES} )



