cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(LIBXS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
if(NOT EXISTS "${LIBXS_VERSION_FILE}")
  message(FATAL_ERROR "Missing ${LIBXS_VERSION_FILE}")
endif()

file(READ "${LIBXS_VERSION_FILE}" LIBXS_VERSION)
string(STRIP "${LIBXS_VERSION}" LIBXS_VERSION)
if(NOT LIBXS_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "Invalid version '${LIBXS_VERSION}' in ${LIBXS_VERSION_FILE}; expected MAJOR.MINOR.PATCH")
endif()

project(libxs
  VERSION "${LIBXS_VERSION}"
  LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckLanguage)

option(LIBXS_FORTRAN "Build Fortran module interface (auto-detected if not set)" OFF)
option(BUILD_TESTING "Build unit tests; requires BLAS" OFF)

check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  set(LIBXS_FORTRAN ON CACHE BOOL "" FORCE)
elseif(LIBXS_FORTRAN)
  message(FATAL_ERROR "LIBXS_FORTRAN requested but no Fortran compiler found")
endif()

if(LIBXS_FORTRAN)
  enable_language(Fortran)
endif()

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXS_FORTRAN         : ${LIBXS_FORTRAN}")

set(LIBXS_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXS_INCLUDE_DIR ${LIBXS_ROOT_DIR}/libxs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_SKIP_TEST_ALL_DEPENDENCY FALSE)
include(CTest)

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

# Re-run on reconfigure whenever version metadata changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LIBXS_VERSION_FILE}"
  "${LIBXS_VERSION_SCRIPT}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(CheckLibraryExists)
set(LIBXS_PUBLIC_LIBRARIES Threads::Threads ${CMAKE_DL_LIBS})
check_library_exists(m sqrt "" LIBXS_HAS_LIBM)
if(LIBXS_HAS_LIBM)
  list(APPEND LIBXS_PUBLIC_LIBRARIES m)
endif()
check_library_exists(rt clock_gettime "" LIBXS_HAS_LIBRT)
if(LIBXS_HAS_LIBRT)
  list(APPEND LIBXS_PUBLIC_LIBRARIES rt)
endif()

set(LIBXS_PUBLIC_INCLUDE_DIRS
  $<BUILD_INTERFACE:${LIBXS_ROOT_DIR}>
  $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libxs>)

file(GLOB LIBXS_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
  ${LIBXS_ROOT_DIR}/src/*.c)

if(LIBXS_FORTRAN)
  list(APPEND LIBXS_SRCS ${LIBXS_INCLUDE_DIR}/libxs.f)
endif()

add_library(libxs ${LIBXS_SRCS})
add_library(libxs::libxs ALIAS libxs)
set_target_properties(libxs PROPERTIES
  OUTPUT_NAME xs
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(libxs PRIVATE LIBXS_BUILD=1)

target_include_directories(libxs
  PUBLIC
    ${LIBXS_PUBLIC_INCLUDE_DIRS}
  PRIVATE
    ${LIBXS_ROOT_DIR}/src)

target_link_libraries(libxs PUBLIC ${LIBXS_PUBLIC_LIBRARIES})

if(LIBXS_FORTRAN)
  set(LIBXS_Fortran_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod)
  set_target_properties(libxs PROPERTIES Fortran_MODULE_DIRECTORY ${LIBXS_Fortran_MODULE_DIR})
  target_include_directories(libxs PUBLIC
    $<BUILD_INTERFACE:${LIBXS_Fortran_MODULE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libxs>)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Pkg-config file
set(PREFIX "${CMAKE_INSTALL_PREFIX}")
set(INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}/libxs")
set(LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
set(PROJECT "libxs")
set(DESCRIPTION "Specialized tensor operations")
set(URL "https://github.com/hfp/libxs/")
set(VERSION "${PROJECT_VERSION}")
if(BUILD_SHARED_LIBS)
  set(LIBS "-L\${libdir} -lxs")
  set(LIBS_PRIVATE "")
else()
  set(LIBS "\${libdir}/libxs.a")
  set(LIBS_PRIVATE "Libs.private: -lpthread -ldl")
  if(LIBXS_HAS_LIBM)
    string(APPEND LIBS_PRIVATE " -lm")
  endif()
  if(LIBXS_HAS_LIBRT)
    string(APPEND LIBS_PRIVATE " -lrt")
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxs.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxs.pc"
  @ONLY
)

# CMake package config files
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libxsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxs
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libxsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# Installation rules
install(TARGETS libxs
  EXPORT libxsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install public headers (picks up the freshly generated libxs_version.h)
install(DIRECTORY ${LIBXS_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxs
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.f"
    PATTERN "*.fi"
    PATTERN "*.F"
    PATTERN "*.mod")

# Install header-only source files (always available)
file(READ "${LIBXS_INCLUDE_DIR}/libxs_source.h" LIBXS_SOURCE_HEADER)
string(REPLACE "../src/" "src/" LIBXS_SOURCE_HEADER "${LIBXS_SOURCE_HEADER}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libxs_source.h" "${LIBXS_SOURCE_HEADER}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libxs_source.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxs)
install(DIRECTORY ${LIBXS_ROOT_DIR}/src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxs/src
  FILES_MATCHING
    PATTERN "*.c"
    PATTERN "*.h")

if(LIBXS_FORTRAN)
  install(DIRECTORY ${LIBXS_Fortran_MODULE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxs
    FILES_MATCHING PATTERN "*.mod")
endif()

# Install pkg-config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxs.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# CMake package config export
install(EXPORT libxsTargets
  FILE libxsTargets.cmake
  NAMESPACE libxs::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxs)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/libxsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/libxsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxs)

# Install documentation into share/libxs
set(LIBXS_SHARE_DIR ${CMAKE_INSTALL_DATADIR}/libxs)

file(GLOB LIBXS_DOC_MDS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxs_*.md")
list(REMOVE_ITEM LIBXS_DOC_MDS
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxs_samples.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxs_scripts.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/libxs_fortran.md")

set(LIBXS_DOC_EXTRA
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/index.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/LICENSE.md")

file(GLOB LIBXS_DOC_PDFS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/documentation/*.pdf")

install(FILES ${LIBXS_DOC_MDS} ${LIBXS_DOC_EXTRA} ${LIBXS_DOC_PDFS}
  DESTINATION ${LIBXS_SHARE_DIR})

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/Makefile.inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/.state.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/.mktmp.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/.flock.sh"
  DESTINATION ${LIBXS_SHARE_DIR})
