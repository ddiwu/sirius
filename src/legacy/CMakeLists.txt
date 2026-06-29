# =============================================================================
# Legacy Sirius code path (gpu_processing table function).
#
# Deleting this directory removes all legacy build support. The main
# CMakeLists.txt detects its presence via EXISTS and skips it when absent.
# =============================================================================

option(ENABLE_LEGACY_SIRIUS
       "Enable legacy Sirius code path (gpu_processing table function)" OFF)

# How many GPUs the legacy magi runtime is built for. Compile-time so the
# barrier / endpoint template sizes are constant. Pass `-DSIRIUS_LEGACY_NUM_GPUS=N`
# at configure time; defaults to the header's #define when unset. The actual
# `target_compile_definitions` happens in the parent CMakeLists.txt after
# `build_static_extension(sirius …)` creates the sirius_extension target.
set(SIRIUS_LEGACY_NUM_GPUS "" CACHE STRING
    "Number of GPUs the legacy magi runtime targets (empty = header default)")

if(ENABLE_LEGACY_SIRIUS)

  # Sources compiled only when the legacy entry point is enabled.
  # cmake-format: off
set(SIRIUS_LEGACY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cpu_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/gpu_expression_executor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/gpu_expression_executor_state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_between.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_case.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_cast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_comparison.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_conjunction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_constant.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_function.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_operator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expression_executor/specializations/gpu_execute_reference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fallback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_columns.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_buffer_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_executor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_meta_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_physical_operator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_query_result.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_materialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_column_data_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_concat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_cte.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_delim_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_dummy_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_empty_result.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_grouped_aggregate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_hash_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_limit.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_nested_loop_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_order.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_partition.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_projection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_result_collector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_strings_matching.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_substring.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_table_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_top_n.cpp
    ${CMAKE_CURRENT_LIST_DIR}/operator/gpu_physical_ungrouped_aggregate.cpp
    PARENT_SCOPE)

set(SIRIUS_LEGACY_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cuda/allocator.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/communication.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/print.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/utils.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/expression_executor/gpu_dispatch_materialize.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/expression_executor/gpu_dispatch_select.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/expression_executor/gpu_dispatch_string.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_aggregate.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_duplicate_elimination.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_groupby.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_join.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_orderby.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/cudf/cudf_utils.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/arbitrary_expression.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/comparison_expression.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/empty_str_check.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/hash_join_inner.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/hash_join_right.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/hash_join_single.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/materialize.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/nested_loop_join.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/strings_matching.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/strlen_from_offsets.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda/operator/substring.cu
)

# ── Optional: Magi-based cross-GPU TPC-H query dispatchers ─────────────────
# When enabled, sirius links the Magi data-plane (NVLink P2P shuffle) as
# its cross-GPU group-by backend for selected TPC-H queries. Magi lives at
# /Magi-Dev (git submodule) — header-only data-plane, but the kernels link
# against system gdrcopy + UCX + numa.
option(ENABLE_MAGI_TPCH
       "Use Magi NVLink-shuffle backend for cross-GPU TPC-H queries" OFF)
if(ENABLE_MAGI_TPCH)
  list(APPEND SIRIUS_LEGACY_CUDA_SOURCES
       ${CMAKE_CURRENT_LIST_DIR}/cuda/magi/magi_runtime.cu
       ${CMAKE_CURRENT_LIST_DIR}/cuda/magi/magi_groupby_runtime.cu
       ${CMAKE_CURRENT_LIST_DIR}/cuda/magi/magi_groupby.cu
       ${CMAKE_CURRENT_LIST_DIR}/cuda/magi/magi_join_runtime.cu
       ${CMAKE_CURRENT_LIST_DIR}/cuda/magi/magi_join.cu)

  # Magi data-plane header tree (data_plane/api, data_plane/queries, ...).
  # CMAKE_SOURCE_DIR is the *duckdb* subdir when the build is driven via
  # the duckdb preset, so we use PROJECT_SOURCE_DIR (= sirius's project
  # root) instead. Normalise the path because BUILD_INTERFACE wrapping
  # in the parent target_include_directories does not handle a path
  # containing `..` correctly — it strips the leading `/` and re-roots
  # everything relative to the target's source dir, producing the famous
  # `-I/home/diwu/sirius/home/diwu/sirius/...` doubled prefix.
  get_filename_component(_magi_root_inc
                         "${PROJECT_SOURCE_DIR}/Magi-Dev/include"
                         ABSOLUTE)
  set(_magi_includes "${_magi_root_inc}")

  # Native libs Magi needs at link time. The single-node Q1 path uses
  # gdrcopy + numa only; UCX is needed for the cross-host network layer
  # (network/ucx.cuh) which Q1 does not include, so we don't depend on it
  # here. Add it back if a future query pulls in network/ucx.cuh.
  #
  # Use the absolute path to libgdrapi.so once we know the prefix, so the
  # dependency propagates to downstream consumers (the duckdb shell that
  # links sirius_extension) without needing to also propagate
  # link_directories — target_link_directories(PRIVATE) is not propagated,
  # and PUBLIC would pollute every transitive consumer's search path.
  set(_magi_link_libs "")
  set(_magi_link_dirs "")

  # gdrcopy detection — try in this order:
  #   1. cmake -DSIRIUS_GDRCOPY_PREFIX=/path/to/gdrcopy
  #   2. environment variable EBROOTGDRCOPY (cluster `module load gdrcopy`)
  #   3. cvmfs fallback for the firlfs / fc10xxx slurm cluster
  #   4. /usr/local (system install)
  set(_magi_gdr_prefix "")
  if(SIRIUS_GDRCOPY_PREFIX)
    set(_magi_gdr_prefix "${SIRIUS_GDRCOPY_PREFIX}")
  elseif(DEFINED ENV{EBROOTGDRCOPY})
    set(_magi_gdr_prefix "$ENV{EBROOTGDRCOPY}")
  elseif(EXISTS "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/gdrcopy/2.3/include/gdrapi.h")
    set(_magi_gdr_prefix
        "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/gdrcopy/2.3")
  elseif(EXISTS "/usr/local/include/gdrapi.h")
    # Headers and libs live directly under /usr/local on a system install.
  else()
    message(FATAL_ERROR
            "ENABLE_MAGI_TPCH=ON but no gdrcopy install found. "
            "Pass -DSIRIUS_GDRCOPY_PREFIX=/path/to/gdrcopy, "
            "module load gdrcopy, or install gdrcopy under /usr/local.")
  endif()

  if(_magi_gdr_prefix)
    list(APPEND _magi_includes "${_magi_gdr_prefix}/include")
    # Find the actual libgdrapi.so file. Some installs use lib, others lib64.
    find_library(_magi_gdrapi_lib gdrapi
                 HINTS "${_magi_gdr_prefix}/lib" "${_magi_gdr_prefix}/lib64"
                 NO_DEFAULT_PATH REQUIRED)
    list(APPEND _magi_link_libs "${_magi_gdrapi_lib}")
    message(STATUS "Magi: gdrcopy at ${_magi_gdr_prefix} (lib=${_magi_gdrapi_lib})")
  endif()

  set(SIRIUS_MAGI_INCLUDE_DIRS    "${_magi_includes}"   PARENT_SCOPE)
  set(SIRIUS_MAGI_LINK_LIBS       "${_magi_link_libs}"  PARENT_SCOPE)
  set(SIRIUS_MAGI_LINK_DIRS       "${_magi_link_dirs}"  PARENT_SCOPE)
  set(SIRIUS_MAGI_COMPILE_DEFINITIONS SIRIUS_ENABLE_MAGI_TPCH PARENT_SCOPE)
  message(STATUS "Magi cross-GPU TPC-H backend is ENABLED")
  message(STATUS "  CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "  PROJECT_SOURCE_DIR       = ${PROJECT_SOURCE_DIR}")
  message(STATUS "  _magi_includes           = ${_magi_includes}")
endif()

set(SIRIUS_LEGACY_CUDA_SOURCES ${SIRIUS_LEGACY_CUDA_SOURCES} PARENT_SCOPE)

set(SIRIUS_LEGACY_CONDITIONAL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/gpu_context.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_physical_plan_generator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_aggregate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_column_data_get.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_comparison_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_cte.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_delim_get.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_delim_join.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_dummy_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_empty_result.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_expression_get.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_get.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_limit.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_order.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_projection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_recursive_cte.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plan/gpu_plan_top_n.cpp
    PARENT_SCOPE)

set(SIRIUS_LEGACY_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../test/cpp/memory_management/test_cpu_cache.cpp
    PARENT_SCOPE)

set(SIRIUS_LEGACY_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/legacy"
    PARENT_SCOPE)
set(SIRIUS_LEGACY_COMPILE_DEFINITIONS SIRIUS_ENABLE_LEGACY PARENT_SCOPE)
message(STATUS "Legacy Sirius code path (gpu_processing) is ENABLED")
else()
  set(SIRIUS_LEGACY_SOURCES "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_CUDA_SOURCES "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_CONDITIONAL_SOURCES "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_TEST_SOURCES "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_INCLUDE_DIR "" PARENT_SCOPE)
  set(SIRIUS_LEGACY_COMPILE_DEFINITIONS "" PARENT_SCOPE)
  message(STATUS "Legacy Sirius code path (gpu_processing) is DISABLED")
endif()
# cmake-format: on
