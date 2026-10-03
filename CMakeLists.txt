cmake_minimum_required(VERSION 3.16)

project(distengine LANGUAGES CXX)

add_library(distengine_core STATIC
    src/engine/version.cpp
    src/engine/request.cpp
)

target_compile_features(distengine_core PUBLIC cxx_std_17)

target_include_directories(distengine_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(test_version
    tests/test_version.cpp
)

target_link_libraries(test_version PRIVATE
    distengine_core
)

enable_testing()
add_test(NAME version_test COMMAND test_version)

add_executable(test_request tests/test_request.cpp)
target_link_libraries(test_request PRIVATE distengine_core)
add_test(NAME request_test COMMAND test_request)
