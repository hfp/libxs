# Native CTest coverage for the GNU Make LIBXS test suite.
#
# The test runner itself is the existing tests/test.sh script, but all
# executables it runs are built by CMake.  This keeps the CTest path
# independent from the Makefile path while preserving the same test list and
# shell-script checks used by `make test`.

find_program(LIBXS_TEST_SHELL NAMES bash sh REQUIRED)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(LIBXS_STATIC_BLAS_LIBRARIES)
set(LIBXS_HAVE_STATIC_BLAS ${BLAS_FOUND})
if(BLAS_FOUND)
  foreach(lib IN LISTS BLAS_LIBRARIES)
    if(IS_ABSOLUTE "${lib}" AND EXISTS "${lib}")
      get_filename_component(lib_dir "${lib}" DIRECTORY)
      get_filename_component(lib_name "${lib}" NAME)
      string(REGEX REPLACE "\\.(so(\\.[0-9]+)*|dylib)$" ".a" static_name "${lib_name}")
      set(static_lib "${lib_dir}/${static_name}")
      if(EXISTS "${static_lib}")
        list(APPEND LIBXS_STATIC_BLAS_LIBRARIES "${static_lib}")
      elseif(lib MATCHES "\\.(so(\\.[0-9]+)*|dylib)$")
        set(LIBXS_HAVE_STATIC_BLAS FALSE)
      else()
        list(APPEND LIBXS_STATIC_BLAS_LIBRARIES "${lib}")
      endif()
    else()
      list(APPEND LIBXS_STATIC_BLAS_LIBRARIES "${lib}")
    endif()
  endforeach()
endif()

set(LIBXS_TEST_BINARY_ROOT "${PROJECT_BINARY_DIR}")
set(LIBXS_TEST_DIR "${LIBXS_TEST_BINARY_ROOT}/tests")
set(LIBXS_SAMPLE_DIR "${LIBXS_TEST_BINARY_ROOT}/samples")
set(LIBXS_TEST_COMMON_INCLUDES
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/libxs"
  "${PROJECT_SOURCE_DIR}/src")
set(LIBXS_SAMPLE_Fortran_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mod")
set(LIBXS_CTEST_PREREQUISITES)

file(MAKE_DIRECTORY
  "${LIBXS_TEST_DIR}"
  "${LIBXS_TEST_BINARY_ROOT}/scripts")

file(COPY
  "${CMAKE_CURRENT_SOURCE_DIR}/test.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/gemm.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/matcpy.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/memcmp.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/ozaki.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/scratch.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/syrk.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/transpose.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/wrap.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/mhd_image.mhd"
  "${CMAKE_CURRENT_SOURCE_DIR}/mhd_image.raw"
  DESTINATION "${LIBXS_TEST_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/scripts/tool_pexec.sh"
  DESTINATION "${LIBXS_TEST_BINARY_ROOT}/scripts")

file(GLOB LIBXS_TEST_RUNNER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
file(COPY ${LIBXS_TEST_RUNNER_SOURCES}
  DESTINATION "${LIBXS_TEST_DIR}")

function(libxs_set_runtime_output target rel_dir output_name)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${LIBXS_TEST_BINARY_ROOT}/${rel_dir}"
    OUTPUT_NAME "${output_name}"
    SUFFIX ".x")
endfunction()

function(libxs_common_target_setup target)
  target_include_directories(${target} PRIVATE ${LIBXS_TEST_COMMON_INCLUDES})
  if(OpenMP_C_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
  endif()
endfunction()

function(libxs_add_c_program target rel_dir output_name)
  add_executable(${target} ${ARGN})
  libxs_common_target_setup(${target})
  target_link_libraries(${target} PRIVATE libxs)
  libxs_set_runtime_output(${target} "${rel_dir}" "${output_name}")
  set(LIBXS_CTEST_PREREQUISITES ${LIBXS_CTEST_PREREQUISITES} ${target} PARENT_SCOPE)
endfunction()

function(libxs_add_fortran_program target rel_dir output_name)
  if(LIBXS_FORTRAN)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE
      ${LIBXS_TEST_COMMON_INCLUDES}
      "${LIBXS_Fortran_MODULE_DIR}"
      "${LIBXS_SAMPLE_Fortran_MODULE_DIR}")
    target_link_libraries(${target} PRIVATE libxs)
    if(BLAS_FOUND)
      target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
    endif()
    if(OpenMP_Fortran_FOUND)
      target_link_libraries(${target} PRIVATE OpenMP::OpenMP_Fortran)
    endif()
    libxs_set_runtime_output(${target} "${rel_dir}" "${output_name}")
    set(LIBXS_CTEST_PREREQUISITES ${LIBXS_CTEST_PREREQUISITES} ${target} PARENT_SCOPE)
  endif()
endfunction()

# C tests discovered by tests/test.sh.
set(LIBXS_HEADER_ONLY_TESTS atomics hash headeronly matdiff memory)
set(LIBXS_C_TESTS
  atomics
  fprint
  hash
  hist
  intrinsics
  malloc
  matdiff
  math
  memory
  mhd
  perm
  registry
  setdiff
  timer)

foreach(test_name IN LISTS LIBXS_C_TESTS)
  set(test_target "libxs_test_${test_name}")
  add_executable(${test_target} "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.c")
  libxs_common_target_setup(${test_target})
  target_compile_definitions(${test_target} PRIVATE LIBXS_BLAS_CONST)
  if(test_name IN_LIST LIBXS_HEADER_ONLY_TESTS)
    target_link_libraries(${test_target} PRIVATE ${LIBXS_PUBLIC_LIBRARIES})
  else()
    target_link_libraries(${test_target} PRIVATE libxs)
  endif()
  libxs_set_runtime_output(${test_target} "tests" "${test_name}")
  list(APPEND LIBXS_CTEST_PREREQUISITES ${test_target})
endforeach()

add_executable(libxs_test_headeronly
  "${CMAKE_CURRENT_SOURCE_DIR}/headeronly.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/headeronly_aux.c")
libxs_common_target_setup(libxs_test_headeronly)
target_compile_definitions(libxs_test_headeronly PRIVATE LIBXS_BLAS_CONST)
target_link_libraries(libxs_test_headeronly PRIVATE ${LIBXS_PUBLIC_LIBRARIES})
libxs_set_runtime_output(libxs_test_headeronly "tests" "headeronly")
list(APPEND LIBXS_CTEST_PREREQUISITES libxs_test_headeronly)

# Sample programs which are built by the GNU Make tests/Makefile before running
# tests/test.sh.  They are either used by shell tests or built as part of the
# existing sample coverage.
libxs_add_c_program(libxs_sample_fprint samples/fprint fprint
  "${PROJECT_SOURCE_DIR}/samples/fprint/fprint.c")

libxs_add_c_program(libxs_sample_rosetta samples/rosetta rosetta
  "${PROJECT_SOURCE_DIR}/samples/rosetta/rosetta.c")

libxs_add_c_program(libxs_sample_setdiff samples/setdiff setdiff
  "${PROJECT_SOURCE_DIR}/samples/setdiff/setdiff.c")

libxs_add_c_program(libxs_sample_shuffle samples/shuffle shuffle
  "${PROJECT_SOURCE_DIR}/samples/shuffle/shuffle.c")

libxs_add_c_program(libxs_sample_sync samples/sync sync
  "${PROJECT_SOURCE_DIR}/samples/sync/sync.c")

libxs_add_c_program(libxs_sample_registry samples/registry registry
  "${PROJECT_SOURCE_DIR}/samples/registry/registry.c")
libxs_add_fortran_program(libxs_sample_registryf samples/registry registryf
  "${PROJECT_SOURCE_DIR}/samples/registry/registry.f")

foreach(name IN ITEMS
    predict_params
    predict_sunspots
    predict_earthquakes
    predict_discharge
    predict_soi
    predict_stock
    predict_crystal)
  libxs_add_c_program("libxs_sample_${name}" samples/predict "${name}"
    "${PROJECT_SOURCE_DIR}/samples/predict/${name}.c")
endforeach()

foreach(name IN ITEMS gemm_strided gemm_batch gemm_groups gemm_index)
  libxs_add_c_program("libxs_sample_${name}" samples/gemm "${name}"
    "${PROJECT_SOURCE_DIR}/samples/gemm/${name}.c")
endforeach()
if(LIBXS_FORTRAN)
  add_library(libxs_sample_jit OBJECT "${PROJECT_SOURCE_DIR}/libxs/libxs_jit.F")
  target_compile_definitions(libxs_sample_jit PRIVATE __LIBXS)
  target_include_directories(libxs_sample_jit PRIVATE
    ${LIBXS_TEST_COMMON_INCLUDES}
    "${LIBXS_Fortran_MODULE_DIR}")
  set_target_properties(libxs_sample_jit PROPERTIES
    Fortran_MODULE_DIRECTORY "${LIBXS_SAMPLE_Fortran_MODULE_DIR}")
  list(APPEND LIBXS_CTEST_PREREQUISITES libxs_sample_jit)
  libxs_add_fortran_program(libxs_sample_gemm_indexf samples/gemm gemm_indexf
    "${PROJECT_SOURCE_DIR}/samples/gemm/gemm_index.f"
    $<TARGET_OBJECTS:libxs_sample_jit>)
endif()

libxs_add_c_program(libxs_sample_memcmp samples/memory memcmp
  "${PROJECT_SOURCE_DIR}/samples/memory/memcmp.c")
libxs_add_c_program(libxs_sample_itrans samples/memory itrans
  "${PROJECT_SOURCE_DIR}/samples/memory/itrans.c")
libxs_add_fortran_program(libxs_sample_memcmpf samples/memory memcmpf
  "${PROJECT_SOURCE_DIR}/samples/memory/memcmp.f")
libxs_add_fortran_program(libxs_sample_matcpyf samples/memory matcpyf
  "${PROJECT_SOURCE_DIR}/samples/memory/matcpy.f")

libxs_add_c_program(libxs_sample_scratch samples/scratch scratch
  "${PROJECT_SOURCE_DIR}/samples/scratch/scratch.c")
libxs_add_fortran_program(libxs_sample_scratchf samples/scratch scratchf
  "${PROJECT_SOURCE_DIR}/samples/scratch/scratch.f")

if(LIBXS_FORTRAN)
  libxs_add_fortran_program(libxs_sample_syrk samples/syrk syrk
    "${PROJECT_SOURCE_DIR}/samples/syrk/syrk.f"
    $<TARGET_OBJECTS:libxs_sample_jit>)
endif()

# Ozaki sample/wrapper tests need BLAS.  If BLAS is not found, the corresponding
# shell tests keep the same skip behavior as the existing scripts because the
# executables are absent.
set(LIBXS_HAVE_OZAKI_TESTS FALSE)
if(BLAS_FOUND AND NOT APPLE)
  set(LIBXS_HAVE_OZAKI_TESTS TRUE)
  set(LIBXS_OZAKI_DIR "${LIBXS_SAMPLE_DIR}/ozaki")
  file(MAKE_DIRECTORY "${LIBXS_OZAKI_DIR}")
  file(COPY
    "${PROJECT_SOURCE_DIR}/samples/ozaki/test-check.sh"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/test-wrap.sh"
    DESTINATION "${LIBXS_OZAKI_DIR}")
  file(WRITE "${LIBXS_OZAKI_DIR}/.state" "BLAS=1\n")

  set(LIBXS_OZAKI_IMPL_SRCS
    "${PROJECT_SOURCE_DIR}/samples/ozaki/ozaki.c"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/ozaki1_int8.c"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/ozaki1_test.c"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/ozaki2_int8.c"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/wrap3m.c"
    "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")

  add_library(libxs_ozaki_impl_d OBJECT ${LIBXS_OZAKI_IMPL_SRCS})
  add_library(libxs_ozaki_impl_s OBJECT ${LIBXS_OZAKI_IMPL_SRCS})
  foreach(target IN ITEMS libxs_ozaki_impl_d libxs_ozaki_impl_s)
    libxs_common_target_setup(${target})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_compile_definitions(${target} PRIVATE __NO_INTRINSICS)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
  target_compile_definitions(libxs_ozaki_impl_s PRIVATE GEMM_REAL_TYPE=float)

  add_library(libxs_ozaki_entry_d OBJECT "${PROJECT_SOURCE_DIR}/samples/ozaki/wrap.c")
  add_library(libxs_ozaki_entry_s OBJECT "${PROJECT_SOURCE_DIR}/samples/ozaki/wrap.c")
  foreach(target IN ITEMS libxs_ozaki_entry_d libxs_ozaki_entry_s)
    libxs_common_target_setup(${target})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_compile_definitions(${target} PRIVATE __NO_INTRINSICS)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
  target_compile_definitions(libxs_ozaki_entry_s PRIVATE GEMM_REAL_TYPE=float)

  add_library(libxs_ozaki_wrap_static STATIC
    $<TARGET_OBJECTS:libxs_ozaki_impl_d>
    $<TARGET_OBJECTS:libxs_ozaki_impl_s>)
  set_target_properties(libxs_ozaki_wrap_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}"
    OUTPUT_NAME wrap)
  target_link_libraries(libxs_ozaki_wrap_static PRIVATE libxs ${BLAS_LIBRARIES})
  list(APPEND LIBXS_CTEST_PREREQUISITES libxs_ozaki_wrap_static)

  add_library(libxs_ozaki_wrap_shared SHARED
    $<TARGET_OBJECTS:libxs_ozaki_impl_d>
    $<TARGET_OBJECTS:libxs_ozaki_impl_s>
    $<TARGET_OBJECTS:libxs_ozaki_entry_d>
    $<TARGET_OBJECTS:libxs_ozaki_entry_s>)
  set_target_properties(libxs_ozaki_wrap_shared PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}"
    OUTPUT_NAME wrap)
  target_link_libraries(libxs_ozaki_wrap_shared PRIVATE libxs ${BLAS_LIBRARIES} ${CMAKE_DL_LIBS})
  list(APPEND LIBXS_CTEST_PREREQUISITES libxs_ozaki_wrap_shared)

  function(libxs_add_ozaki_driver target output_name)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm.c")
    libxs_common_target_setup(${target})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_compile_definitions(${target} PRIVATE __NO_INTRINSICS)
    target_link_libraries(${target} PRIVATE libxs ${BLAS_LIBRARIES})
    libxs_set_runtime_output(${target} "samples/ozaki" "${output_name}")
  endfunction()

  libxs_add_ozaki_driver(libxs_sample_dgemm_blas dgemm-blas)
  target_sources(libxs_sample_dgemm_blas PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")

  libxs_add_ozaki_driver(libxs_sample_sgemm_blas sgemm-blas)
  target_sources(libxs_sample_sgemm_blas PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_sgemm_blas PRIVATE GEMM_REAL_TYPE=float)

  libxs_add_ozaki_driver(libxs_sample_zgemm_blas zgemm-blas)
  target_sources(libxs_sample_zgemm_blas PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_zgemm_blas PRIVATE GEMM_COMPLEX)

  libxs_add_ozaki_driver(libxs_sample_cgemm_blas cgemm-blas)
  target_sources(libxs_sample_cgemm_blas PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_cgemm_blas PRIVATE GEMM_COMPLEX GEMM_REAL_TYPE=float)

  function(libxs_add_ozaki_wrap_driver target output_name)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm.c")
    libxs_common_target_setup(${target})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_compile_definitions(${target} PRIVATE __NO_INTRINSICS)
    target_link_libraries(${target} PRIVATE libxs_ozaki_wrap_static libxs ${LIBXS_STATIC_BLAS_LIBRARIES})
    target_link_options(${target} PRIVATE
      "-Wl,--wrap=dgemm_"
      "-Wl,--wrap=sgemm_"
      "-Wl,--wrap=zgemm_"
      "-Wl,--wrap=cgemm_")
    libxs_set_runtime_output(${target} "samples/ozaki" "${output_name}")
    set(LIBXS_CTEST_PREREQUISITES ${LIBXS_CTEST_PREREQUISITES} ${target} PARENT_SCOPE)
  endfunction()

  if(LIBXS_HAVE_STATIC_BLAS)
    libxs_add_ozaki_wrap_driver(libxs_sample_dgemm_wrap dgemm-wrap)
    libxs_add_ozaki_wrap_driver(libxs_sample_sgemm_wrap sgemm-wrap)
    target_compile_definitions(libxs_sample_sgemm_wrap PRIVATE GEMM_REAL_TYPE=float)
    libxs_add_ozaki_wrap_driver(libxs_sample_zgemm_wrap zgemm-wrap)
    target_compile_definitions(libxs_sample_zgemm_wrap PRIVATE GEMM_COMPLEX)
    libxs_add_ozaki_wrap_driver(libxs_sample_cgemm_wrap cgemm-wrap)
    target_compile_definitions(libxs_sample_cgemm_wrap PRIVATE GEMM_COMPLEX GEMM_REAL_TYPE=float)
  else()
    message(STATUS "Static BLAS library not found; Ozaki static --wrap executables will be skipped")
  endif()
endif()

set(LIBXS_TEST_ENTRIES
  atomics.c
  fprint.c
  hash.c
  headeronly.c
  hist.c
  intrinsics.c
  malloc.c
  matdiff.c
  math.c
  memory.c
  mhd.c
  perm.c
  registry.c
  setdiff.c
  timer.c
  gemm.sh
  matcpy.sh
  memcmp.sh
  ozaki.sh
  scratch.sh
  syrk.sh
  transpose.sh
  wrap.sh)

foreach(test_entry IN LISTS LIBXS_TEST_ENTRIES)
  get_filename_component(test_name "${test_entry}" NAME_WE)
  add_test(NAME "libxs-${test_name}"
    COMMAND "${LIBXS_TEST_SHELL}" "${LIBXS_TEST_DIR}/test.sh" "${test_entry}")
  set_tests_properties("libxs-${test_name}" PROPERTIES
    WORKING_DIRECTORY "${LIBXS_TEST_DIR}"
    LABELS libxs)
endforeach()

add_custom_target(libxs_ctest_prerequisites
  DEPENDS ${LIBXS_CTEST_PREREQUISITES})
add_custom_target(libxs-check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS libxs_ctest_prerequisites
  USES_TERMINAL)
