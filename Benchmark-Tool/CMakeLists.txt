cmake_minimum_required(VERSION 3.16...3.27
  FATAL_ERROR
)

project(Benchmark-Tool
  VERSION 1.0 
  LANGUAGES CXX 
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose type of build (Debug or Release)" FORCE
  )
endif()

add_executable(app General-Usage/general-usage.cpp)

target_compile_options(app PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-O3 -Wall -Wpedantic>
  $<$<CXX_COMPILER_ID:Clang>:-O3 -Wall -Wpedantic>
)
