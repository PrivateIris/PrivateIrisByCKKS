# CMakeLists.stansalone.txt Standalone CMake configuration for installation

cmake_minimum_required(VERSION 3.14)
project(HEavenStandaloneInstall)

# --- Build examples using installed files ---
set(EXAMPLES_DIR "${CMAKE_SOURCE_DIR}")
file(GLOB EXAMPLE_SOURCES "${EXAMPLES_DIR}/*.cpp")

include_directories("${CMAKE_SOURCE_DIR}/include")
link_directories("${CMAKE_SOURCE_DIR}/lib")

find_package(ZLIB REQUIRED)

foreach(example_src ${EXAMPLE_SOURCES})
  get_filename_component(example_name ${example_src} NAME_WE)
  add_executable(${example_name} ${example_src})
  target_include_directories(${example_name}
                             PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(${example_name} PRIVATE worldcoin)
  target_link_libraries(${example_name} PRIVATE ZLIB::ZLIB)
endforeach()

message(
  STATUS
    "Standalone install CMake configuration loaded. Examples will be built using installed headers and libraries."
)
