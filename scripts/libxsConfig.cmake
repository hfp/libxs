# lib/cmake/libxs or lib/<target>/cmake/libxs
get_filename_component(_libxs_libdir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(_libxs_prefix "${_libxs_libdir}/.." ABSOLUTE)
if(NOT EXISTS "${_libxs_prefix}/include/libxs/libxs.h"
   AND NOT EXISTS "${_libxs_prefix}/libxs/libxs.h")
  get_filename_component(_libxs_prefix "${_libxs_libdir}/../.." ABSOLUTE)
endif()
set(LIBXS_OPENMP @LIBXS_OPENMP@)

set(_libxs_suffixes_save "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()
find_library(
  LIBXS_LIBRARY
  NAMES xs
  HINTS "${_libxs_libdir}"
  NO_DEFAULT_PATH)
if(NOT LIBXS_LIBRARY)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxs_suffixes_save}")
  find_library(
    LIBXS_LIBRARY
    NAMES xs
    HINTS "${_libxs_libdir}"
    NO_DEFAULT_PATH)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxs_suffixes_save}")
unset(_libxs_suffixes_save)
# pin "latest" (lib/ symlinks) to the folder of the actual build (lib/<target>)
if(LIBXS_LIBRARY)
  get_filename_component(_libxs_real "${LIBXS_LIBRARY}" REALPATH)
  get_filename_component(_libxs_real "${_libxs_real}" DIRECTORY)
  get_filename_component(_libxs_name "${LIBXS_LIBRARY}" NAME)
  if(EXISTS "${_libxs_real}/${_libxs_name}")
    set(_libxs_libdir "${_libxs_real}")
    set(LIBXS_LIBRARY
        "${_libxs_real}/${_libxs_name}"
        CACHE FILEPATH "Path to a library." FORCE)
  endif()
  unset(_libxs_name)
  unset(_libxs_real)
endif()
find_path(
  LIBXS_INCLUDE_DIR
  NAMES libxs/libxs.h
  HINTS "${_libxs_prefix}/include" "${_libxs_prefix}"
  NO_DEFAULT_PATH)

if(LIBXS_LIBRARY AND LIBXS_INCLUDE_DIR)
  if(NOT TARGET libxs::libxs)
    find_package(Threads QUIET)
    if(LIBXS_OPENMP)
      find_package(OpenMP REQUIRED COMPONENTS C)
    endif()
    set(_incdirs "${LIBXS_INCLUDE_DIR}")
    if(EXISTS "${_libxs_libdir}/libxs.mod")
      list(APPEND _incdirs "${_libxs_libdir}")
    endif()
    # Fortran sources consumers compile themselves (libxs_jit.F)
    if(EXISTS "${LIBXS_INCLUDE_DIR}/libxs/libxs.mod"
       OR EXISTS "${LIBXS_INCLUDE_DIR}/libxs/libxs_jit.F")
      list(APPEND _incdirs "${LIBXS_INCLUDE_DIR}/libxs")
    endif()
    add_library(libxs::libxs UNKNOWN IMPORTED)
    set_target_properties(
      libxs::libxs PROPERTIES IMPORTED_LOCATION "${LIBXS_LIBRARY}"
                              INTERFACE_INCLUDE_DIRECTORIES "${_incdirs}")
    if(TARGET Threads::Threads)
      set_property(
        TARGET libxs::libxs
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads)
    endif()
    if(LIBXS_OPENMP)
      set_property(
        TARGET libxs::libxs
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_C)
    endif()
    # same link interface as the CMake-exported target, i.e. consumers of a GNU
    # Make build resolve sqrt/pow, dlopen, and clock_gettime alike
    include(CheckLibraryExists)
    set(_libxs_libs ${CMAKE_DL_LIBS})
    check_library_exists(m sqrt "" LIBXS_HAS_LIBM)
    if(LIBXS_HAS_LIBM)
      list(APPEND _libxs_libs m)
    endif()
    check_library_exists(rt clock_gettime "" LIBXS_HAS_LIBRT)
    if(LIBXS_HAS_LIBRT)
      list(APPEND _libxs_libs rt)
    endif()
    if(_libxs_libs)
      set_property(
        TARGET libxs::libxs
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES ${_libxs_libs})
    endif()
    unset(_libxs_libs)
    unset(_incdirs)
  else()
    set_property(TARGET libxs::libxs PROPERTY IMPORTED_LOCATION
                                              "${LIBXS_LIBRARY}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libxs DEFAULT_MSG LIBXS_LIBRARY
                                  LIBXS_INCLUDE_DIR)
unset(_libxs_prefix)
unset(_libxs_libdir)
unset(LIBXS_OPENMP)
