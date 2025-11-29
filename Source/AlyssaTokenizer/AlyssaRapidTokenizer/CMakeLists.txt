cmake_minimum_required(VERSION 3.15)
project(AlyssaRapidTokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(Third-party/Pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Third-party/NlohmannJSON)

add_library(alyssa_rapid_tokenizer MODULE Source/AlyssaRapidTokenizer.cpp Source/PythonBindings.cpp)

target_link_libraries(alyssa_rapid_tokenizer PRIVATE pybind11::module)

set_target_properties(alyssa_rapid_tokenizer PROPERTIES PREFIX "")

if(WIN32)
    set_target_properties(alyssa_rapid_tokenizer PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(alyssa_rapid_tokenizer PROPERTIES SUFFIX ".so")
endif()
