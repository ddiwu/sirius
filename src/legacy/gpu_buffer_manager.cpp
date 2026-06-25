/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu_buffer_manager.hpp"

#include "config.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "helper/types.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "utils.hpp"

#include <cstdio>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>

// Local alias for kSiriusLegacyNumGpus (declared in the header) so existing
// code paths can keep using NUM_GPUS without churn.
#define NUM_GPUS kSiriusLegacyNumGpus

namespace duckdb {

namespace {

inline void set_current_gpu_resource(gpu_pool_memory_resource& mr)
{
  cudf::set_current_device_resource_ref(rmm::device_async_resource_ref{mr});
}

inline gpu_pool_memory_resource* make_gpu_pool_memory_resource(
  rmm::mr::cuda_memory_resource* upstream,
  std::size_t initial_pool_size,
  std::size_t maximum_pool_size)
{
#if (RMM_VERSION_MAJOR > 26) || (RMM_VERSION_MAJOR == 26 && RMM_VERSION_MINOR >= 6)
  return new gpu_pool_memory_resource(
    rmm::device_async_resource_ref{*upstream}, initial_pool_size, maximum_pool_size);
#else
  return new gpu_pool_memory_resource(upstream, initial_pool_size, maximum_pool_size);
#endif
}

}  // namespace
using sirius::AggregationType;
using sirius::OrderByType;

template int16_t* GPUBufferManager::customCudaMalloc<int16_t>(size_t size, int gpu, bool caching);

template int* GPUBufferManager::customCudaMalloc<int>(size_t size, int gpu, bool caching);

template uint64_t* GPUBufferManager::customCudaMalloc<uint64_t>(size_t size, int gpu, bool caching);

template int64_t* GPUBufferManager::customCudaMalloc<int64_t>(size_t size, int gpu, bool caching);

template uint32_t* GPUBufferManager::customCudaMalloc<uint32_t>(size_t size, int gpu, bool caching);

template uint8_t* GPUBufferManager::customCudaMalloc<uint8_t>(size_t size, int gpu, bool caching);

template float* GPUBufferManager::customCudaMalloc<float>(size_t size, int gpu, bool caching);

template double* GPUBufferManager::customCudaMalloc<double>(size_t size, int gpu, bool caching);

template char* GPUBufferManager::customCudaMalloc<char>(size_t size, int gpu, bool caching);

template bool* GPUBufferManager::customCudaMalloc<bool>(size_t size, int gpu, bool caching);

template __int128_t* GPUBufferManager::customCudaMalloc<__int128_t>(size_t size,
                                                                    int gpu,
                                                                    bool caching);

template duckdb_string_type* GPUBufferManager::customCudaMalloc<duckdb_string_type>(size_t size,
                                                                                    int gpu,
                                                                                    bool caching);

template pointer_and_key* GPUBufferManager::customCudaMalloc<pointer_and_key>(size_t size,
                                                                              int gpu,
                                                                              bool caching);

template string_group_by_metadata_type*
GPUBufferManager::customCudaMalloc<string_group_by_metadata_type>(size_t size,
                                                                  int gpu,
                                                                  bool caching);

template void** GPUBufferManager::customCudaMalloc<void*>(size_t size, int gpu, bool caching);

template string_group_by_record_type*
GPUBufferManager::customCudaMalloc<string_group_by_record_type>(size_t size, int gpu, bool caching);

template string_top_n_record_type* GPUBufferManager::customCudaMalloc<string_top_n_record_type>(
  size_t size, int gpu, bool caching);

template uint8_t** GPUBufferManager::customCudaMalloc<uint8_t*>(size_t size, int gpu, bool caching);

template uint64_t** GPUBufferManager::customCudaMalloc<uint64_t*>(size_t size,
                                                                  int gpu,
                                                                  bool caching);

template uint32_t** GPUBufferManager::customCudaMalloc<uint32_t*>(size_t size,
                                                                  int gpu,
                                                                  bool caching);

template int32_t** GPUBufferManager::customCudaMalloc<int32_t*>(size_t size, int gpu, bool caching);

template int64_t** GPUBufferManager::customCudaMalloc<int64_t*>(size_t size, int gpu, bool caching);

template double** GPUBufferManager::customCudaMalloc<double*>(size_t size, int gpu, bool caching);

template int* GPUBufferManager::customCudaHostAlloc<int>(size_t size);

template int64_t* GPUBufferManager::customCudaHostAlloc<int64_t>(size_t size);

template uint64_t* GPUBufferManager::customCudaHostAlloc<uint64_t>(size_t size);

template uint32_t* GPUBufferManager::customCudaHostAlloc<uint32_t>(size_t size);

template uint8_t* GPUBufferManager::customCudaHostAlloc<uint8_t>(size_t size);

template float* GPUBufferManager::customCudaHostAlloc<float>(size_t size);

template double* GPUBufferManager::customCudaHostAlloc<double>(size_t size);

template char* GPUBufferManager::customCudaHostAlloc<char>(size_t size);

template bool* GPUBufferManager::customCudaHostAlloc<bool>(size_t size);

template string_t* GPUBufferManager::customCudaHostAlloc<string_t>(size_t size);

template AggregationType* GPUBufferManager::customCudaHostAlloc<AggregationType>(size_t size);

template OrderByType* GPUBufferManager::customCudaHostAlloc<OrderByType>(size_t size);

template ScanDataType* GPUBufferManager::customCudaHostAlloc<ScanDataType>(size_t size);

template CompareType* GPUBufferManager::customCudaHostAlloc<CompareType>(size_t size);

template int** GPUBufferManager::customCudaHostAlloc<int*>(size_t size);

template int64_t** GPUBufferManager::customCudaHostAlloc<int64_t*>(size_t size);

template uint64_t** GPUBufferManager::customCudaHostAlloc<uint64_t*>(size_t size);

template uint8_t** GPUBufferManager::customCudaHostAlloc<uint8_t*>(size_t size);

template uint32_t** GPUBufferManager::customCudaHostAlloc<uint32_t*>(size_t size);

template float** GPUBufferManager::customCudaHostAlloc<float*>(size_t size);

template double** GPUBufferManager::customCudaHostAlloc<double*>(size_t size);

template char** GPUBufferManager::customCudaHostAlloc<char*>(size_t size);

template string_t** GPUBufferManager::customCudaHostAlloc<string_t*>(size_t size);

template ConstantFilter** GPUBufferManager::customCudaHostAlloc<ConstantFilter*>(size_t size);

GPUBufferManager::GPUBufferManager(size_t cache_size_per_gpu,
                                   size_t processing_size_per_gpu,
                                   size_t processing_size_per_cpu)
  : cache_size_per_gpu(cache_size_per_gpu),
    processing_size_per_gpu(processing_size_per_gpu),
    processing_size_per_cpu(processing_size_per_cpu)
{
  SIRIUS_LOG_INFO(
    "Initializing GPU buffer manager with params: Use Pin - {}, Use Pin For Caching - {}, GPU "
    "Cache Size - {}, GPU Processing Size - {}, CPU Processing Size - {}",
    Config::USE_PIN_MEM_FOR_CPU_PROCESSING,
    Config::USE_PIN_MEM_FOR_CACHING,
    cache_size_per_gpu,
    processing_size_per_gpu,
    processing_size_per_cpu);
  gpuCache             = new uint8_t*[NUM_GPUS];
  cpuCache             = new uint8_t*[NUM_GPUS];
  gpuProcessing        = new uint8_t*[NUM_GPUS];
  cpuProcessing        = Config::USE_PIN_MEM_FOR_CPU_PROCESSING
                           ? allocatePinnedCPUMemory(processing_size_per_cpu)
                           : allocatePageableCPUMemory(processing_size_per_cpu);
  gpuProcessingPointer = new size_t[NUM_GPUS];
  gpuCachingPointer    = new size_t[NUM_GPUS];
  cpuCachingPointer    = new size_t[NUM_GPUS];
  cpuProcessingPointer = 0;
  available_gpu_cache_size.resize(NUM_GPUS);
  tables_per_gpu.resize(NUM_GPUS);

  // Phase 2: per-GPU processing pools. Each pool lives on its own device so
  // kernels launched from a thread bound to GPU g allocate intermediate
  // buffers on GPU g (no cross-GPU pointers).
  cuda_mr_per_gpu.assign(NUM_GPUS, nullptr);
  mr_per_gpu.assign(NUM_GPUS, nullptr);
  for (int g = 0; g < NUM_GPUS; ++g) {
    cudaSetDevice(g);
    cuda_mr_per_gpu[g] = new rmm::mr::cuda_memory_resource();
    mr_per_gpu[g]      = make_gpu_pool_memory_resource(
      cuda_mr_per_gpu[g], processing_size_per_gpu, processing_size_per_gpu);
    SIRIUS_LOG_INFO("Allocated processing size {} on GPU {}", processing_size_per_gpu, g);
  }
  cudaSetDevice(0);
  // Backwards-compatible aliases: code that hasn't been taught about the
  // per-thread current GPU still sees GPU 0's pool.
  cuda_mr = cuda_mr_per_gpu[0];
  mr      = mr_per_gpu[0];
  set_current_gpu_resource(*mr);
  allocation_table.resize(NUM_GPUS);
  locked_allocation_table.resize(NUM_GPUS);
  rmm_stored_buffers.resize(NUM_GPUS);

  for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
    if (Config::USE_PIN_MEM_FOR_CACHING) {
      gpuCache[gpu]                 = callCudaHostAlloc<uint8_t>(cache_size_per_gpu, 1);
      cpuCache[gpu]                 = nullptr;
      available_gpu_cache_size[gpu] = cache_size_per_gpu;
      SIRIUS_LOG_INFO("Allocated cache size {} using pinned host memory", cache_size_per_gpu);
    } else {
      // We cannot allocate exactly all free memory using `cudaMalloc()`
      size_t free_gpu_mem_size = getFreeGPUMemorySize(gpu) * 0.99;
      if (free_gpu_mem_size >= cache_size_per_gpu) {
        gpuCache[gpu]                 = callCudaMalloc<uint8_t>(cache_size_per_gpu, gpu);
        cpuCache[gpu]                 = nullptr;
        available_gpu_cache_size[gpu] = cache_size_per_gpu;
        SIRIUS_LOG_INFO("Allocated cache size {} in GPU 0", cache_size_per_gpu);
      } else {
        gpuCache[gpu] = callCudaMalloc<uint8_t>(free_gpu_mem_size, gpu);
        cpuCache[gpu] = allocatePinnedCPUMemory(cache_size_per_gpu - free_gpu_mem_size);
        available_gpu_cache_size[gpu] = free_gpu_mem_size;
        SIRIUS_LOG_INFO("Allocated cache size {} for GPU 0 ({} in GPU, {} in CPU)",
                        cache_size_per_gpu,
                        free_gpu_mem_size,
                        cache_size_per_gpu - free_gpu_mem_size);
      }
    }

    gpuProcessingPointer[gpu] = 0;
    gpuProcessing[gpu]        = nullptr;
    gpuCachingPointer[gpu]    = 0;
    cpuCachingPointer[gpu]    = 0;
  }

  warmup_gpu();
}

GPUBufferManager::~GPUBufferManager()
{
  for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
    if (Config::USE_PIN_MEM_FOR_CACHING) {
      freePinnedCPUMemory(gpuCache[gpu]);
    } else {
      callCudaFree<uint8_t>(gpuCache[gpu], gpu);
      if (cpuCache[gpu] != nullptr) { freePinnedCPUMemory(cpuCache[gpu]); }
    }
    if (gpuProcessing[gpu] != nullptr) {
      mr->deallocate(rmm::cuda_stream_view{},
                     static_cast<void*>(gpuProcessing[gpu]),
                     processing_size_per_gpu,
                     rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
  }
  Config::USE_PIN_MEM_FOR_CPU_PROCESSING ? freePinnedCPUMemory(cpuProcessing)
                                         : freePageableCPUMemory(cpuProcessing);
  delete[] gpuCache;
  delete[] cpuCache;
  delete[] gpuProcessing;
  delete[] gpuProcessingPointer;
  delete[] gpuCachingPointer;
  delete[] cpuCachingPointer;
  rmm_stored_buffers.clear();
  for (int g = 0; g < NUM_GPUS; ++g) {
    cudaSetDevice(g);
    delete mr_per_gpu[g];
    delete cuda_mr_per_gpu[g];
  }
  cudaSetDevice(0);
  // `mr` and `cuda_mr` were aliases — already freed above.
  mr      = nullptr;
  cuda_mr = nullptr;
}

void GPUBufferManager::set_gpu_for_thread(int g)
{
  auto err = cudaSetDevice(g);
  if (err != cudaSuccess) {
    SIRIUS_LOG_ERROR("set_gpu_for_thread({}): cudaSetDevice failed: {}", g, cudaGetErrorString(err));
  }
  sirius_current_gpu = g;
  set_current_gpu_resource(*mr_per_gpu[g]);
}

void GPUBufferManager::ResetBuffer()
{
  // Called single-threaded between queries, but take the lock defensively so
  // the table/buffer mutations here can never overlap a stray free.
  std::lock_guard<std::mutex> lk(alloc_mutex);
  for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
    SIRIUS_LOG_DEBUG("Resetting buffer for GPU {}", gpu);
    cudaSetDevice(gpu);
    set_current_gpu_resource(*mr_per_gpu[gpu]);
    auto* gpu_mr              = mr_per_gpu[gpu];
    gpuProcessingPointer[gpu] = 0;
    // write a program to free all allocation in the allocation table
    for (auto it = allocation_table[gpu].begin(); it != allocation_table[gpu].end(); ++it) {
      auto ptr  = it->first;
      auto size = it->second;
      if (ptr != nullptr) {
        gpu_mr->deallocate(
          rmm::cuda_stream_view{}, (void*)ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }
    }
    allocation_table[gpu].clear();
    if (!allocation_table[gpu].empty()) {
      throw InvalidInputException("Allocation table is not empty");
    }
    for (auto it = locked_allocation_table[gpu].begin(); it != locked_allocation_table[gpu].end();
         ++it) {
      auto ptr  = it->first;
      auto size = it->second;
      if (ptr != nullptr) {
        gpu_mr->deallocate(
          rmm::cuda_stream_view{}, (void*)ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }
    }
    locked_allocation_table[gpu].clear();
    if (!locked_allocation_table[gpu].empty()) {
      throw InvalidInputException("Locked allocation table is not empty");
    }
    rmm_stored_buffers[gpu].clear();  // per-GPU; outer vector keeps its NUM_GPUS slots
    // SIRIUS_LOG_DEBUG("pool size {}", mr->pool_size());

    // size_t allocated_size = mr->pool_size();
    // SIRIUS_LOG_DEBUG("Allocating {} bytes", allocated_size);
    // void* ptr = mr->allocate(allocated_size);
    // mr->deallocate(ptr, allocated_size);
  }
  cpuProcessingPointer = 0;
  for (auto& tables : tables_per_gpu) {
    for (auto it = tables.begin(); it != tables.end(); it++) {
      shared_ptr<GPUIntermediateRelation> table = it->second;
      for (int col = 0; col < table->columns.size(); col++) {
        if (table->columns[col] != nullptr) {
          table->columns[col]->row_ids      = nullptr;
          table->columns[col]->row_id_count = 0;
        }
      }
    }
  }
}

void GPUBufferManager::ResetCache()
{
  SIRIUS_LOG_DEBUG("Resetting cache");
  for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
    gpuCachingPointer[gpu] = 0;
    cpuCachingPointer[gpu] = 0;
  }
  for (auto& tables : tables_per_gpu) {
    for (auto it = tables.begin(); it != tables.end(); it++) {
      shared_ptr<GPUIntermediateRelation> table = it->second;
      for (int col = 0; col < table->columns.size(); col++) {
        table->columns[col] = nullptr;
      }
      table->column_names.clear();
      table->column_names.resize(table->column_count);
    }
  }
}

template <typename T>
T* GPUBufferManager::customCudaMalloc(size_t size, int gpu, bool caching)
{
  size_t alloc = (size * sizeof(T));
  // always ensure that it aligns with RMM's CUDA allocation alignment
  //  size_t alignment = alignof(double);
  size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
  alloc += (alignment - (alloc % alignment)) % alignment;
  if (caching) {
    // Caching path keeps the explicit `gpu` argument (used by the scan to
    // write per-GPU partitions of cached tables).
    size_t start = __atomic_fetch_add(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
    T* ptr       = nullptr;
    if (start + alloc <= available_gpu_cache_size[gpu]) {
      ptr = reinterpret_cast<T*>(gpuCache[gpu] + start);
    } else {
      __atomic_fetch_sub(&gpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
      start = __atomic_fetch_add(&cpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
      if (start + alloc > cache_size_per_gpu - available_gpu_cache_size[gpu]) {
        __atomic_fetch_sub(&cpuCachingPointer[gpu], alloc, __ATOMIC_RELAXED);
        throw InvalidInputException("Out of caching memory");
      }
      ptr = reinterpret_cast<T*>(cpuCache[gpu] + start);
    }
    if (reinterpret_cast<uintptr_t>(ptr) % alignof(double) != 0) {
      throw InvalidInputException("Memory is not properly aligned");
    }
    return ptr;
  } else {
    // Processing path ignores the (legacy) `gpu` argument and routes to the
    // per-thread current GPU's pool. The thread's caller (executor worker)
    // is expected to have called set_gpu_for_thread() already.
    int t_gpu  = sirius_current_gpu;
    auto* gmr  = mr_per_gpu[t_gpu];
    void* ptr  = gmr->allocate(rmm::cuda_stream_view{}, alloc, rmm::CUDA_ALLOCATION_ALIGNMENT);
    if (ptr == nullptr) throw InvalidInputException("Pointer is nullptr");
    // RMM allocate above is left outside the lock (RMM is thread-safe); only
    // the std::map bookkeeping is serialized, because another GPU's worker may
    // be scanning allocation_table[t_gpu] in customCudaFree's cross-GPU path.
    std::lock_guard<std::mutex> lk(alloc_mutex);
    if (allocation_table[t_gpu].find(ptr) != allocation_table[t_gpu].end()) {
      throw InvalidInputException("Pointer already exists in allocation table");
    }
    allocation_table[t_gpu][ptr] = alloc;
    return reinterpret_cast<T*>(ptr);
  }
}

void GPUBufferManager::lockAllocation(void* ptr, int gpu)
{
  // move entries from the allocation table to the locked table
  std::lock_guard<std::mutex> lk(alloc_mutex);
  auto it = allocation_table[gpu].find(ptr);
  if (it != allocation_table[gpu].end()) {
    // SIRIUS_LOG_DEBUG("Locking Pointer {}", static_cast<void*>(ptr));
    locked_allocation_table[gpu][ptr] = it->second;
    allocation_table[gpu].erase(it);
  }
}

void GPUBufferManager::customCudaFree(uint8_t* ptr, int gpu)
{
  // The legacy callers pass `gpu` as the cache GPU (matters for the
  // gpuCache/cpuCache range checks), but processing-pool allocations are
  // tracked by the per-thread current GPU (where they were recorded at
  // alloc time in customCudaMalloc). Search both:
  //   1) the caller's `gpu` table  (covers single-thread legacy usage)
  //   2) the current thread's GPU table  (covers Phase 2 multi-GPU)
  // Whichever owns the pointer wins.
  if (ptr == nullptr) { return; }
  // Cache range check is per-GPU (caching slabs are owned per-GPU).
  for (int g = 0; g < static_cast<int>(mr_per_gpu.size()); ++g) {
    if (ptr >= gpuCache[g] && ptr < gpuCache[g] + available_gpu_cache_size[g]) { return; }
    if (cpuCache[g] != nullptr && ptr >= cpuCache[g] &&
        ptr < cpuCache[g] + cache_size_per_gpu - available_gpu_cache_size[g]) {
      return;
    }
  }

  // Everything below reads/writes the allocation tables and rmm_stored_buffers,
  // which other GPUs' worker threads mutate concurrently — serialize it.
  std::lock_guard<std::mutex> lk(alloc_mutex);

  auto try_dealloc = [&](int g) -> bool {
    auto it = allocation_table[g].find(reinterpret_cast<void*>(ptr));
    if (it != allocation_table[g].end()) {
      mr_per_gpu[g]->deallocate(
        rmm::cuda_stream_view{}, (void*)ptr, it->second, rmm::CUDA_ALLOCATION_ALIGNMENT);
      allocation_table[g].erase(it);
      return true;
    }
    return false;
  };

  // Own GPU first (cheapest common case). Cross-GPU frees do happen (a cudf
  // join / cache-load intermediate allocated under one current-GPU and freed
  // by another worker), so the fallback scans below are real — they just must
  // run under the lock above.
  const int cur = sirius_current_gpu;
  if (try_dealloc(cur)) return;
  if (cur == gpu && locked_allocation_table[cur].find(reinterpret_cast<void*>(ptr)) !=
                      locked_allocation_table[cur].end()) {
    return;  // locked on own GPU, do not free
  }
  for (auto& buf : rmm_stored_buffers[cur]) {
    if (ptr == reinterpret_cast<uint8_t*>(buf->data())) { return; }  // own-GPU stored buffer
  }

  // Anything below touches ANOTHER GPU's bookkeeping (cross-GPU free). That is
  // only legitimate from a single-threaded phase (result consolidation / reset).
  // Instrumented: if this ever fires while the per-GPU workers run concurrently
  // it would be a data race — the [XGPUFREE] marker lets us confirm it doesn't.
  if (gpu != cur && try_dealloc(gpu)) {
    std::fprintf(stderr, "[XGPUFREE] explicit cur=%d gpu=%d\n", cur, gpu);
    return;
  }
  for (int g = 0; g < static_cast<int>(allocation_table.size()); ++g) {
    if (g == cur || g == gpu) continue;
    if (try_dealloc(g)) {
      std::fprintf(stderr, "[XGPUFREE] alloc-scan cur=%d found=%d\n", cur, g);
      return;
    }
  }
  for (int g = 0; g < static_cast<int>(mr_per_gpu.size()); ++g) {
    if (g == cur) continue;
    if (locked_allocation_table[g].find(reinterpret_cast<void*>(ptr)) !=
        locked_allocation_table[g].end()) {
      std::fprintf(stderr, "[XGPUFREE] locked-scan cur=%d found=%d\n", cur, g);
      return;  // locked, do not free
    }
  }
  for (int g = 0; g < static_cast<int>(rmm_stored_buffers.size()); ++g) {
    if (g == cur) continue;
    for (auto& buf : rmm_stored_buffers[g]) {
      if (ptr == reinterpret_cast<uint8_t*>(buf->data())) {
        std::fprintf(stderr, "[XGPUFREE] stored-scan cur=%d found=%d\n", cur, g);
        return;
      }
    }
  }

  SIRIUS_LOG_DEBUG("Invalid Pointer {}", static_cast<void*>(ptr));
  // A VALID device pointer that is in no sirius allocation table is owned by cudf
  // (allocated via the RMM resource inside a cudf join/groupby, not customCudaMalloc).
  // cudf frees it through its own resource when the result table is destroyed, so
  // customCudaFree must NOT throw or double-free here — just skip it. Only a truly
  // wild/unregistered pointer (lookup error or not device memory) is a real bug.
  {
    cudaPointerAttributes a{};
    cudaError_t           e = cudaPointerGetAttributes(&a, reinterpret_cast<void*>(ptr));
    cudaGetLastError();
    if (e == cudaSuccess && a.type == cudaMemoryTypeDevice) {
      static int warned = 0;
      if (warned++ < 3)
        SIRIUS_LOG_DEBUG("customCudaFree: skipping cudf-owned device ptr not in sirius table");
      return;
    }
    std::fprintf(stderr, "[FREE-WILD] ptr=%p cur_gpu=%d sirius_current_gpu=%d perr=%d ptr_dev=%d ptr_type=%d\n",
                 reinterpret_cast<void*>(ptr), cur, sirius_current_gpu, static_cast<int>(e),
                 a.device, static_cast<int>(a.type));
    std::fflush(stderr);
  }
  throw InvalidInputException("Pointer not found in allocation table");
}

template <typename T>
T* GPUBufferManager::customCudaHostAlloc(size_t size)
{
  size_t alloc = (size * sizeof(T));
  size_t start = __atomic_fetch_add(&cpuProcessingPointer, alloc, __ATOMIC_RELAXED);
  assert((start + alloc) < processing_size_per_cpu);
  if (start + alloc >= processing_size_per_cpu) {
    throw InvalidInputException("Out of CPU memory");
  }
  return reinterpret_cast<T*>(cpuProcessing + start);
}

void GPUBufferManager::createTableAndColumnInGPU(Catalog& catalog,
                                                 ClientContext& context,
                                                 string table_name,
                                                 string column_name,
                                                 int gpu)
{
  SIRIUS_LOG_DEBUG(
    "CreateTable and Column called for table {} and col {} on GPU {}",
    table_name,
    column_name,
    gpu);
  TableCatalogEntry& table =
    catalog.GetEntry(context, CatalogType::TABLE_ENTRY, DEFAULT_SCHEMA, table_name)
      .Cast<TableCatalogEntry>();
  auto column_names = table.GetColumns().GetColumnNames();
  auto& constraints = table.GetConstraints();
  vector<size_t> unique_columns;

  for (auto& constraint : constraints) {
    if (constraint->type == ConstraintType::UNIQUE) {
      auto& pk = constraint->Cast<UniqueConstraint>();
      if (pk.HasIndex()) {
        SIRIUS_LOG_DEBUG("Unique constraint on index {}", pk.GetIndex().index);
        for (auto& col : pk.GetColumnNames()) {
          SIRIUS_LOG_DEBUG("Unique constraint on column {}", col);
        }
        unique_columns.push_back(pk.GetIndex().index);
      } else {
        for (auto& col : pk.GetColumnNames()) {
          SIRIUS_LOG_DEBUG("Unique constraint on column {}", col);
        }
      }
    }
  }

  // finding column_name in column_names
  // convert column_name to uppercase
  string up_column_name = column_name;
  // when caching table, it has to be exactly the same as the column name in the table (case
  // sensitive)
  transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
  if (find(column_names.begin(), column_names.end(), column_name) != column_names.end()) {
    // convert table_name to uppercase
    size_t column_id     = table.GetColumnIndex(column_name, false).index;
    string up_table_name = table_name;
    transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
    createTable(up_table_name, table.GetTypes().size(), gpu);
    GPUColumnType column_type =
      convertLogicalTypeToColumnType(table.GetColumn(column_name).GetType());
    SIRIUS_LOG_DEBUG("Creating column {}", up_column_name);
    createColumn(up_table_name, up_column_name, column_type, column_id, unique_columns, gpu);
  } else {
    throw InvalidInputException("Column does not exists");
  }
  SIRIUS_LOG_DEBUG("Table and column created in GPU {}", gpu);
}

void GPUBufferManager::createTable(string up_table_name, size_t column_count, int gpu)
{
  // we will update the length later
  // check if table already exists
  SIRIUS_LOG_DEBUG(
    "Crate Table called for table {} with {} cols on GPU {}", up_table_name, column_count, gpu);
  auto& tables = tables_per_gpu[gpu];
  if (tables.find(up_table_name) == tables.end()) {
    tables[up_table_name]        = make_shared_ptr<GPUIntermediateRelation>(column_count);
    tables[up_table_name]->names = up_table_name;
  }
}

bool GPUBufferManager::checkIfColumnCached(string table_name, string column_name, int gpu)
{
  string up_column_name = column_name;
  string up_table_name  = table_name;
  transform(up_table_name.begin(), up_table_name.end(), up_table_name.begin(), ::toupper);
  transform(up_column_name.begin(), up_column_name.end(), up_column_name.begin(), ::toupper);
  auto& tables  = tables_per_gpu[gpu];
  auto table_it = tables.find(up_table_name);
  if (table_it == tables.end()) { return false; }
  const auto& table = table_it->second;
  auto column_it    = find(table->column_names.begin(), table->column_names.end(), up_column_name);
  if (column_it == table->column_names.end()) { return false; }
  return true;
}

void GPUBufferManager::createColumn(string up_table_name,
                                    string up_column_name,
                                    GPUColumnType column_type,
                                    size_t column_id,
                                    vector<size_t> unique_columns,
                                    int gpu)
{
  shared_ptr<GPUIntermediateRelation> table = tables_per_gpu[gpu][up_table_name];
  table->column_names[column_id]            = up_column_name;
  if (find(unique_columns.begin(), unique_columns.end(), column_id) != unique_columns.end()) {
    table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr, nullptr);
    table->columns[column_id]->is_unique = true;
  } else {
    table->columns[column_id] = make_shared_ptr<GPUColumn>(0, column_type, nullptr, nullptr);
    table->columns[column_id]->is_unique = false;
  }
}

}  // namespace duckdb
