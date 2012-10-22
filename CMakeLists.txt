cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(IPyraNet)

#set(Boost_USE_STATIC_LIBS ON)

find_package(OpenCV COMPONENTS core highgui REQUIRED)
#find_package(Boost COMPONENTS system thread REQUIRED)

include_directories(${OPENNI_INCLUDE_DIR})
#include_directories(${Boost_INCLUDE_DIRS})
#link_directories(${Boost_LIBRARY_DIRS})

set(IPyraNet_SOURCE_FILES 
    source/main.cxx
	)
	
add_executable (IPyraNet ${IPyraNet_SOURCE_FILES})

#target_link_libraries(IPyraNet ${Boost_LIBRARIES})
target_link_libraries(IPyraNet ${OpenCV_LIBS})
