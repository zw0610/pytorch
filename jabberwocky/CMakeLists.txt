cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(jabberwocky)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Torch REQUIRED)

add_executable(array_addition array_addition.cpp)
target_link_libraries(array_addition "${TORCH_LIBRARIES}")
set_property(TARGET array_addition PROPERTY CXX_STANDARD 11)

add_executable(inplace_array_addition inplace_array_addition.cpp)
target_link_libraries(inplace_array_addition "${TORCH_LIBRARIES}")
set_property(TARGET inplace_array_addition PROPERTY CXX_STANDARD 11)

add_executable(torch_addition torch_addition.cpp)
target_link_libraries(torch_addition "${TORCH_LIBRARIES}")
set_property(TARGET torch_addition PROPERTY CXX_STANDARD 11)

add_executable(hash_benchmark hash_benchmark.cpp)
target_link_libraries(hash_benchmark "${TORCH_LIBRARIES}")
set_property(TARGET hash_benchmark PROPERTY CXX_STANDARD 11)

add_executable(jabberwocky jabberwocky.cpp)
target_link_libraries(jabberwocky "${TORCH_LIBRARIES}")
set_property(TARGET jabberwocky PROPERTY CXX_STANDARD 11)
