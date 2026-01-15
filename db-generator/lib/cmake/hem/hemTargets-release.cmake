#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hem::hem" for configuration "Release"
set_property(TARGET hem::hem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hem::hem PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhem.a"
  )

list(APPEND _cmake_import_check_targets hem::hem )
list(APPEND _cmake_import_check_files_for_hem::hem "${_IMPORT_PREFIX}/lib/libhem.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
