# Samples are also needed by the shell tests.
find_package(BLAS REQUIRED)

set(LIBXS_SAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBXS_SAMPLE_Fortran_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mod")

function(libxs_set_runtime_output target rel_dir output_name)
  # Shell tests expect the same layout with single- and multi-config generators.
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${rel_dir}/$<0:>"
               OUTPUT_NAME "${output_name}"
               SUFFIX ".x")
endfunction()

function(libxs_common_target_setup target)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PRIVATE libxs)
endfunction()

function(libxs_add_c_program target rel_dir output_name)
  add_executable(${target} ${ARGN})
  libxs_common_target_setup(${target})
  libxs_set_runtime_output(${target} "${rel_dir}" "${output_name}")
endfunction()

function(libxs_add_fortran_program target rel_dir output_name)
  if(LIBXS_FORTRAN)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE libxs OpenMP::OpenMP_Fortran
                                            ${BLAS_LIBRARIES})
    libxs_set_runtime_output(${target} "${rel_dir}" "${output_name}")
  endif()
endfunction()

# Keep the supplied corpora and normalizer data beside their executables.
file(
  GLOB
  LIBXS_SAMPLE_DATA
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/predict/*.csv"
  "${CMAKE_CURRENT_SOURCE_DIR}/predict/*.tsv"
  "${CMAKE_CURRENT_SOURCE_DIR}/predict/*.dat"
  "${CMAKE_CURRENT_SOURCE_DIR}/converse/*.rules"
  "${CMAKE_CURRENT_SOURCE_DIR}/converse/*.norms")
foreach(data_file IN LISTS LIBXS_SAMPLE_DATA)
  file(RELATIVE_PATH rel_file "${CMAKE_CURRENT_SOURCE_DIR}" "${data_file}")
  configure_file("${data_file}" "${CMAKE_CURRENT_BINARY_DIR}/${rel_file}"
                 COPYONLY)
endforeach()
configure_file(predict/predict_params.sh predict/predict_params.sh COPYONLY)

set(LIBXS_STATIC_BLAS_LIBRARIES)
set(LIBXS_HAVE_STATIC_BLAS ${BLAS_FOUND})
if(BLAS_FOUND)
  foreach(lib IN LISTS BLAS_LIBRARIES)
    if(IS_ABSOLUTE "${lib}" AND EXISTS "${lib}")
      get_filename_component(lib_dir "${lib}" DIRECTORY)
      get_filename_component(lib_name "${lib}" NAME)
      string(REGEX REPLACE "\\.(so(\\.[0-9]+)*|dylib)$" ".a" static_name
                           "${lib_name}")
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

libxs_add_c_program(
  libxs_sample_bracket samples/bracket bracket
  "${CMAKE_CURRENT_SOURCE_DIR}/bracket/bracket.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/bracket/bracket_predict.c")
target_link_libraries(libxs_sample_bracket PRIVATE ${BLAS_LIBRARIES})

libxs_add_c_program(libxs_sample_pack samples/pack pack
                    "${CMAKE_CURRENT_SOURCE_DIR}/pack/pack.c")
if(LIBXS_FORTRAN)
  target_sources(libxs_sample_pack
                 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pack/vpack.f")
  target_compile_definitions(libxs_sample_pack
                             PRIVATE $<$<COMPILE_LANGUAGE:C>:PACK_FORTRAN>)
  target_link_libraries(libxs_sample_pack PRIVATE OpenMP::OpenMP_Fortran)
endif()

set(LIBXS_CONVERSE_DIR "${PROJECT_SOURCE_DIR}/samples/converse")

add_library(libxs_sample_converse_core OBJECT
            "${LIBXS_CONVERSE_DIR}/converse.c")
add_library(
  libxs_sample_converse_qa OBJECT "${LIBXS_CONVERSE_DIR}/converse_qa.c"
                                  "${LIBXS_CONVERSE_DIR}/converse_recomb.c")
add_library(
  libxs_sample_converse_lm OBJECT "${LIBXS_CONVERSE_DIR}/converse_lm.c"
                                  "${LIBXS_CONVERSE_DIR}/converse_hier.c")

foreach(target IN ITEMS libxs_sample_converse_core libxs_sample_converse_qa
                        libxs_sample_converse_lm)
  libxs_common_target_setup(${target})
endforeach()

libxs_add_c_program(
  libxs_sample_converse
  samples/converse
  converse
  "${LIBXS_CONVERSE_DIR}/main_all.c"
  $<TARGET_OBJECTS:libxs_sample_converse_core>
  $<TARGET_OBJECTS:libxs_sample_converse_qa>
  $<TARGET_OBJECTS:libxs_sample_converse_lm>)
libxs_add_c_program(
  libxs_sample_converse_qa_exe
  samples/converse
  converse-qa
  "${LIBXS_CONVERSE_DIR}/main_qa.c"
  $<TARGET_OBJECTS:libxs_sample_converse_core>
  $<TARGET_OBJECTS:libxs_sample_converse_qa>)
libxs_add_c_program(
  libxs_sample_converse_lm_exe
  samples/converse
  converse-lm
  "${LIBXS_CONVERSE_DIR}/main_lm.c"
  $<TARGET_OBJECTS:libxs_sample_converse_core>
  $<TARGET_OBJECTS:libxs_sample_converse_lm>)

libxs_add_c_program(
  libxs_sample_converse_norm samples/converse converse-norm
  "${LIBXS_CONVERSE_DIR}/main_norm.c" "${LIBXS_CONVERSE_DIR}/converse_norm.c"
  $<TARGET_OBJECTS:libxs_sample_converse_core>)

libxs_add_c_program(libxs_sample_summarize samples/converse summarize
                    "${PROJECT_SOURCE_DIR}/samples/converse/summarize.c")

libxs_add_c_program(libxs_sample_fprint samples/fprint fprint
                    "${PROJECT_SOURCE_DIR}/samples/fprint/fprint.c")

libxs_add_c_program(libxs_sample_rosetta samples/rosetta rosetta
                    "${PROJECT_SOURCE_DIR}/samples/rosetta/rosetta.c")

libxs_add_c_program(libxs_sample_setdiff samples/setdiff setdiff
                    "${PROJECT_SOURCE_DIR}/samples/setdiff/setdiff.c")

libxs_add_c_program(libxs_sample_shuffle samples/shuffle shuffle
                    "${PROJECT_SOURCE_DIR}/samples/shuffle/shuffle.c")

libxs_add_c_program(libxs_sample_sync_lock samples/sync sync_lock
                    "${PROJECT_SOURCE_DIR}/samples/sync/sync_lock.c")

# tests/sync.sh runs this one and skips without it, so leaving it out of the
# CTest path would leave that test always skipping and looking like a pass
libxs_add_c_program(libxs_sample_sync_barrier samples/sync sync_barrier
                    "${PROJECT_SOURCE_DIR}/samples/sync/sync_barrier.c")

libxs_add_c_program(libxs_sample_registry samples/registry registry
                    "${PROJECT_SOURCE_DIR}/samples/registry/registry.c")
libxs_add_fortran_program(libxs_sample_registryf samples/registry registryf
                          "${PROJECT_SOURCE_DIR}/samples/registry/registry.f")

foreach(
  name IN
  ITEMS predict_params
        predict_sunspots
        predict_earthquakes
        predict_discharge
        predict_soi
        predict_stock
        predict_crystal
        predict_ett
        predict_prob
        predict_dist
        predict_bits
        predict_higgs)
  libxs_add_c_program("libxs_sample_${name}" samples/predict "${name}"
                      "${PROJECT_SOURCE_DIR}/samples/predict/${name}.c")
endforeach()

foreach(name IN ITEMS gemm_strided gemm_batch gemm_groups gemm_index)
  libxs_add_c_program("libxs_sample_${name}" samples/gemm "${name}"
                      "${PROJECT_SOURCE_DIR}/samples/gemm/${name}.c")
endforeach()
if(LIBXS_FORTRAN)
  add_library(libxs_sample_jit OBJECT "${PROJECT_SOURCE_DIR}/libxs/libxs_jit.F")
  target_link_libraries(libxs_sample_jit PUBLIC libxs OpenMP::OpenMP_Fortran)
  target_compile_definitions(libxs_sample_jit PRIVATE __LIBXS)
  target_include_directories(libxs_sample_jit
                             PUBLIC "${LIBXS_SAMPLE_Fortran_MODULE_DIR}")
  set_target_properties(
    libxs_sample_jit PROPERTIES Fortran_MODULE_DIRECTORY
                                "${LIBXS_SAMPLE_Fortran_MODULE_DIR}")
  libxs_add_fortran_program(libxs_sample_gemm_indexf samples/gemm gemm_indexf
                            "${PROJECT_SOURCE_DIR}/samples/gemm/gemm_index.f")
  target_link_libraries(libxs_sample_gemm_indexf PRIVATE libxs_sample_jit)
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

libxs_add_c_program(libxs_sample_paircnt_dd samples/spatial paircnt_dd
                    "${PROJECT_SOURCE_DIR}/samples/spatial/paircnt_dd.c")

libxs_add_c_program(libxs_sample_tokenizer samples/tokenizer tokenizer
                    "${PROJECT_SOURCE_DIR}/samples/tokenizer/tokenizer.c")

if(LIBXS_FORTRAN)
  libxs_add_fortran_program(libxs_sample_syrk samples/syrk syrk
                            "${PROJECT_SOURCE_DIR}/samples/syrk/syrk.f")
  target_link_libraries(libxs_sample_syrk PRIVATE libxs_sample_jit)
endif()

# Ozaki wrappers are currently built only on non-Apple platforms.
if(NOT APPLE)
  set(LIBXS_OZAKI_DIR "${LIBXS_SAMPLE_DIR}/ozaki")
  file(MAKE_DIRECTORY "${LIBXS_OZAKI_DIR}")
  foreach(script IN ITEMS test-check.sh test-wrap.sh test-mhd.sh)
    configure_file("ozaki/${script}" "${LIBXS_OZAKI_DIR}/${script}" COPYONLY)
  endforeach()
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
    target_include_directories(${target}
                               PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
  target_compile_definitions(libxs_ozaki_impl_s PRIVATE GEMM_REAL_TYPE=float)

  add_library(libxs_ozaki_entry_d OBJECT
              "${PROJECT_SOURCE_DIR}/samples/ozaki/wrap.c")
  add_library(libxs_ozaki_entry_s OBJECT
              "${PROJECT_SOURCE_DIR}/samples/ozaki/wrap.c")
  foreach(target IN ITEMS libxs_ozaki_entry_d libxs_ozaki_entry_s)
    libxs_common_target_setup(${target})
    target_include_directories(${target}
                               PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
  target_compile_definitions(libxs_ozaki_entry_s PRIVATE GEMM_REAL_TYPE=float)

  add_library(
    libxs_ozaki_wrap_static STATIC $<TARGET_OBJECTS:libxs_ozaki_impl_d>
                                   $<TARGET_OBJECTS:libxs_ozaki_impl_s>)
  set_target_properties(
    libxs_ozaki_wrap_static
    PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}/$<0:>" OUTPUT_NAME
                                                                   wrap)
  target_link_libraries(libxs_ozaki_wrap_static PRIVATE libxs ${BLAS_LIBRARIES})

  add_library(
    libxs_ozaki_wrap_shared SHARED
    $<TARGET_OBJECTS:libxs_ozaki_impl_d> $<TARGET_OBJECTS:libxs_ozaki_impl_s>
    $<TARGET_OBJECTS:libxs_ozaki_entry_d> $<TARGET_OBJECTS:libxs_ozaki_entry_s>)
  set_target_properties(
    libxs_ozaki_wrap_shared
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}/$<0:>"
               RUNTIME_OUTPUT_DIRECTORY "${LIBXS_OZAKI_DIR}/$<0:>"
               OUTPUT_NAME wrap)
  target_link_libraries(libxs_ozaki_wrap_shared PRIVATE libxs ${BLAS_LIBRARIES}
                                                        ${CMAKE_DL_LIBS})

  # Keep symbols from static dependencies internal to the BLAS interposer.
  include(CheckCSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES "-Wl,--exclude-libs,ALL")
  check_c_source_compiles("int main(void) { return 0; }"
                          LIBXS_LINKER_HAS_EXCLUDE_LIBS)
  cmake_pop_check_state()
  if(LIBXS_LINKER_HAS_EXCLUDE_LIBS)
    target_link_options(libxs_ozaki_wrap_shared PRIVATE
                        "-Wl,--exclude-libs,ALL")
  endif()

  function(libxs_add_ozaki_driver target output_name)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm.c")
    libxs_common_target_setup(${target})
    target_include_directories(${target}
                               PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_link_libraries(${target} PRIVATE libxs ${BLAS_LIBRARIES})
    libxs_set_runtime_output(${target} "samples/ozaki" "${output_name}")
  endfunction()

  libxs_add_ozaki_driver(libxs_sample_dgemm_blas dgemm-blas)
  target_sources(libxs_sample_dgemm_blas
                 PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")

  libxs_add_ozaki_driver(libxs_sample_sgemm_blas sgemm-blas)
  target_sources(libxs_sample_sgemm_blas
                 PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_sgemm_blas
                             PRIVATE GEMM_REAL_TYPE=float)

  libxs_add_ozaki_driver(libxs_sample_zgemm_blas zgemm-blas)
  target_sources(libxs_sample_zgemm_blas
                 PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_zgemm_blas PRIVATE GEMM_COMPLEX)

  libxs_add_ozaki_driver(libxs_sample_cgemm_blas cgemm-blas)
  target_sources(libxs_sample_cgemm_blas
                 PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm-print.c")
  target_compile_definitions(libxs_sample_cgemm_blas
                             PRIVATE GEMM_COMPLEX GEMM_REAL_TYPE=float)

  function(libxs_add_ozaki_wrap_driver target output_name)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/samples/ozaki/gemm.c")
    libxs_common_target_setup(${target})
    target_include_directories(${target}
                               PRIVATE "${PROJECT_SOURCE_DIR}/samples/ozaki")
    target_link_libraries(${target} PRIVATE libxs_ozaki_wrap_static libxs
                                            ${LIBXS_STATIC_BLAS_LIBRARIES})
    target_link_options(
      ${target} PRIVATE "-Wl,--wrap=dgemm_" "-Wl,--wrap=sgemm_"
      "-Wl,--wrap=zgemm_" "-Wl,--wrap=cgemm_")
    libxs_set_runtime_output(${target} "samples/ozaki" "${output_name}")
  endfunction()

  if(LIBXS_HAVE_STATIC_BLAS)
    libxs_add_ozaki_wrap_driver(libxs_sample_dgemm_wrap dgemm-wrap)
    libxs_add_ozaki_wrap_driver(libxs_sample_sgemm_wrap sgemm-wrap)
    target_compile_definitions(libxs_sample_sgemm_wrap
                               PRIVATE GEMM_REAL_TYPE=float)
    libxs_add_ozaki_wrap_driver(libxs_sample_zgemm_wrap zgemm-wrap)
    target_compile_definitions(libxs_sample_zgemm_wrap PRIVATE GEMM_COMPLEX)
    libxs_add_ozaki_wrap_driver(libxs_sample_cgemm_wrap cgemm-wrap)
    target_compile_definitions(libxs_sample_cgemm_wrap
                               PRIVATE GEMM_COMPLEX GEMM_REAL_TYPE=float)
  else()
    message(
      STATUS
        "Static BLAS library not found; Ozaki static --wrap executables will be skipped"
    )
  endif()
endif()
