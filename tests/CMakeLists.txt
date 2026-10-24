# Use the GNU Make suite's shell runner with executables built by CMake.
find_program(LIBXS_TEST_SHELL NAMES bash REQUIRED)

set(LIBXS_C_TESTS
    fprint
    gemm_dispatch
    hash
    headeronly
    hist
    intrinsics
    malloc
    matdiff
    math
    memory
    mhd
    mhd_png
    mix
    ngram
    pack
    perm
    perm_sort
    predict_blend
    predict_calib
    predict_coord
    predict_hknn_load
    predict_missing
    predict_neighbors
    predict_prob
    predict_reach
    predict_serial
    predict_weights
    registry
    setdiff
    str
    sync_atomics
    sync_barrier
    sync_ring
    sync_singleton
    timer
    token)
set(LIBXS_HEADER_ONLY_TESTS
    hash
    headeronly
    matdiff
    memory
    mix
    pack
    str
    sync_atomics
    sync_barrier
    sync_ring
    sync_singleton)
if(LIBXS_BUILD_SAMPLES)
  set(LIBXS_SHELL_TESTS
      gemm
      matcpy
      memcmp
      ozaki
      predict
      scratch
      sync
      syrk
      transpose
      wrap)
endif()

set(LIBXS_TEST_ENTRIES)
foreach(test_name IN LISTS LIBXS_C_TESTS)
  set(test_target "libxs_test_${test_name}")
  add_executable(${test_target} "${test_name}.c")
  if(test_name STREQUAL "headeronly")
    target_sources(${test_target} PRIVATE headeronly_aux.c)
  endif()
  target_include_directories(
    ${test_target}
    PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/libxs"
            "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(${test_target} PRIVATE LIBXS_BLAS_CONST)
  if(test_name IN_LIST LIBXS_HEADER_ONLY_TESTS)
    target_link_libraries(${test_target} PRIVATE ${LIBXS_PUBLIC_LIBRARIES})
  else()
    target_link_libraries(${test_target} PRIVATE libxs)
  endif()
  set_target_properties(
    ${test_target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<0:>"
               OUTPUT_NAME "${test_name}"
               SUFFIX ".x")
  # test.sh without arguments discovers C tests from these files.
  configure_file("${test_name}.c" "${test_name}.c" COPYONLY)
  list(APPEND LIBXS_TEST_ENTRIES "${test_name}.c")
endforeach()

configure_file(test.sh test.sh COPYONLY)
foreach(test_name IN LISTS LIBXS_SHELL_TESTS)
  configure_file("${test_name}.sh" "${test_name}.sh" COPYONLY)
  list(APPEND LIBXS_TEST_ENTRIES "${test_name}.sh")
endforeach()
foreach(data_file IN ITEMS mhd_image.mhd mhd_image.raw)
  configure_file("${data_file}" "${data_file}" COPYONLY)
endforeach()
configure_file("${PROJECT_SOURCE_DIR}/scripts/tool_pexec.sh"
               "${PROJECT_BINARY_DIR}/scripts/tool_pexec.sh" COPYONLY)

foreach(test_entry IN LISTS LIBXS_TEST_ENTRIES)
  get_filename_component(test_name "${test_entry}" NAME_WE)
  add_test(NAME "libxs-${test_name}"
           COMMAND "${LIBXS_TEST_SHELL}" "${CMAKE_CURRENT_BINARY_DIR}/test.sh"
                   "${test_entry}")
  set_tests_properties(
    "libxs-${test_name}" PROPERTIES WORKING_DIRECTORY
                                    "${CMAKE_CURRENT_BINARY_DIR}" LABELS libxs)
endforeach()
