cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(libxs
  VERSION 0.0.0
  LANGUAGES C)

option(LIBXS_HEADER_ONLY "Header-only usage (no library target)" OFF)
option(LIBXS_SHARED      "Build shared library instead of static" OFF)
option(LIBXS_FORTRAN     "Build Fortran module interface" OFF)

if(LIBXS_FORTRAN)
  set(LIBXS_HEADER_ONLY OFF CACHE BOOL "" FORCE)
  enable_language(Fortran)
endif()

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXS_HEADER_ONLY     : ${LIBXS_HEADER_ONLY}")
message(STATUS "  LIBXS_SHARED          : ${LIBXS_SHARED}")
message(STATUS "  LIBXS_FORTRAN         : ${LIBXS_FORTRAN}")

set(LIBXS_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXS_INCLUDE_DIR ${LIBXS_ROOT_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generate include/libxs_version.h
set(LIBXS_VERSION_HEADER "${LIBXS_INCLUDE_DIR}/libxs_version.h")
set(LIBXS_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tool_version.sh")

if(NOT EXISTS "${LIBXS_VERSION_SCRIPT}")
  message(FATAL_ERROR "Missing ${LIBXS_VERSION_SCRIPT}")
endif()

find_program(SH_EXECUTABLE NAMES sh bash REQUIRED)

# -1 means "emit full C header" (see tool_version.sh)
execute_process(
  COMMAND "${SH_EXECUTABLE}" "${LIBXS_VERSION_SCRIPT}" libxs -1
  OUTPUT_FILE "${LIBXS_VERSION_HEADER}"
  RESULT_VARIABLE LIBXS_VERSION_RC
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT LIBXS_VERSION_RC EQUAL 0)
  message(FATAL_ERROR "tool_version.sh failed with code ${LIBXS_VERSION_RC}")
endif()

# Re-run on reconfigure whenever the script changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LIBXS_VERSION_SCRIPT}")

if(LIBXS_HEADER_ONLY)
  add_library(libxs INTERFACE)
  target_include_directories(libxs INTERFACE
    $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)
  add_library(libxs::libxs ALIAS libxs)
else()
  file(GLOB LIBXS_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
    ${LIBXS_ROOT_DIR}/src/*.c)

  if(LIBXS_FORTRAN)
    list(APPEND LIBXS_SRCS ${LIBXS_INCLUDE_DIR}/libxs.f)
  endif()

  if(LIBXS_SHARED)
    add_library(libxs SHARED ${LIBXS_SRCS})
  else()
    add_library(libxs STATIC ${LIBXS_SRCS})
  endif()
  add_library(libxs::libxs ALIAS libxs)
  set_target_properties(libxs PROPERTIES OUTPUT_NAME xs)

  target_compile_definitions(libxs PRIVATE LIBXS_BUILD=1)

  target_include_directories(libxs
    PUBLIC
      $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${LIBXS_ROOT_DIR}/src)

  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(libxs PUBLIC Threads::Threads)
  target_link_libraries(libxs PUBLIC ${CMAKE_DL_LIBS})

  include(CheckLibraryExists)
  check_library_exists(m sqrt "" LIBXS_HAS_LIBM)
  if(LIBXS_HAS_LIBM)
    target_link_libraries(libxs PUBLIC m)
  endif()
  check_library_exists(rt clock_gettime "" LIBXS_HAS_LIBRT)
  if(LIBXS_HAS_LIBRT)
    target_link_libraries(libxs PUBLIC rt)
  endif()

  if(LIBXS_FORTRAN)
    set(LIBXS_Fortran_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod)
    set_target_properties(libxs PROPERTIES Fortran_MODULE_DIRECTORY ${LIBXS_Fortran_MODULE_DIR})
    target_include_directories(libxs PUBLIC
      $<BUILD_INTERFACE:${LIBXS_Fortran_MODULE_DIR}>
      $<INSTALL_INTERFACE:include>)
  endif()
endif()

# Pkg-config file
if(NOT LIBXS_HEADER_ONLY)
  set(LIBS_PRIVATE "")
  if(NOT LIBXS_SHARED)
    set(LIBS_PRIVATE "Libs.private: -lpthread -ldl")
    if(LIBXS_HAS_LIBM)
      string(APPEND LIBS_PRIVATE " -lm")
    endif()
    if(LIBXS_HAS_LIBRT)
      string(APPEND LIBS_PRIVATE " -lrt")
    endif()
  endif()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libxs.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libxs.pc"
    @ONLY
  )
endif()

# Installation rules
include(GNUInstallDirs)

if(LIBXS_HEADER_ONLY)
  install(TARGETS libxs
    EXPORT libxsTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
  install(TARGETS libxs
    EXPORT libxsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Install public headers (picks up the freshly generated libxs_version.h)
install(DIRECTORY ${LIBXS_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.f"
    PATTERN "*.mod")

if(LIBXS_FORTRAN)
  install(DIRECTORY ${LIBXS_Fortran_MODULE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.mod")
endif()

# Install pkg-config files
if(NOT LIBXS_HEADER_ONLY)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libxs.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  )
endif()

# CMake package config export
install(EXPORT libxsTargets
  FILE libxsTargets.cmake
  NAMESPACE libxs::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxs)
