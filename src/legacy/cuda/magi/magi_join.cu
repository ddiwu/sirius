// magi_join.cu — operator-facing wrapper for the magi shuffle join.
//
// Mirrors magi_groupby.cu's Run: derive the key recipe, build the ColPack
// inputs for both sides + the probe-payload table, drive the runtime, and emit
// the joined rows back as GPUColumns. See magi_join.hpp for the contract.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "legacy/operator/magi_join.hpp"
#include "legacy/operator/magi_distributed_join.hpp"
#include "legacy/operator/magi_distributed_groupby.hpp"  // KeyKind, TableSize, N_SLOTS_*
#include "legacy/gpu_buffer_manager.hpp"

#include "data_plane/ops/col_pack.cuh"  // KeyFieldEntry, KeyFieldKind

namespace duckdb {
namespace magi_join_op {

namespace {

using magi_ops::KeyFieldEntry;
using magi_ops::KeyFieldKind;
using duckdb::magi_generic::JoinPayloadEntry;

// ── Derive the fixed-width key recipe (INT32/INT64 only for v1) ─────────────
// Same packing rule as magi_groupby's DeriveKeyShape: walk keys in order, give
// each its byte range, pick the key word width from the total. src_col_idx is
// the per-kind pool index (BuildJoinInputs assigns i_cols / i64_cols in the
// same order). ≤4B → INT32 key word; ≤8B → UINT64.
bool DeriveKeyShape(const std::vector<shared_ptr<GPUColumn>>& keys,
                    magi_generic::KeyKind&                     out_kind,
                    std::vector<KeyFieldEntry>&                out_fields)
{
  out_fields.clear();
  if (keys.empty()) return false;
  int byte_off = 0, i32_idx = 0, i64_idx = 0;
  for (auto& k : keys) {
    auto id = k->data_wrapper.type.id();
    if (id == GPUColumnTypeId::INT32) {
      out_fields.push_back({KeyFieldKind::INT32, (int8_t)i32_idx++, (int8_t)byte_off, 4});
      byte_off += 4;
    } else if (id == GPUColumnTypeId::INT64) {
      out_fields.push_back({KeyFieldKind::INT64, (int8_t)i64_idx++, (int8_t)byte_off, 8});
      byte_off += 8;
    } else {
      return false;  // v1: fixed-width INT keys only
    }
  }
  if (byte_off > 8) return false;  // v1 instantiates INT32/UINT64 only
  out_kind = (byte_off <= 4) ? magi_generic::KeyKind::INT32
                             : magi_generic::KeyKind::UINT64;
  return true;
}

// ── Build a PerGpuJoinInputs (ColPack) from key + payload columns ───────────
// Key columns → i_cols / i64_cols (key order, matches DeriveKeyShape).
// Payload columns → i64_agg_cols (INT64/DECIMAL) / d_cols (FLOAT64), and each
// gets a JoinPayloadEntry so the pack/emit agree on where it lives. row_validity
// is taken from the key's validity_mask (filtered-out rows skipped).
magi_generic::PerGpuJoinInputs
BuildJoinInputs(const std::vector<shared_ptr<GPUColumn>>& keys,
                const std::vector<shared_ptr<GPUColumn>>& payload,
                std::vector<JoinPayloadEntry>&            out_payload_tbl)
{
  magi_generic::PerGpuJoinInputs in{};
  in.n_rows = keys.empty() ? 0 : keys[0]->column_length;
  in.cols   = {};

  int i_idx = 0, i64_idx = 0;
  for (auto& k : keys) {
    auto id = k->data_wrapper.type.id();
    if (id == GPUColumnTypeId::INT32)
      in.cols.i_cols[i_idx++] = reinterpret_cast<const int32_t*>(k->data_wrapper.data);
    else if (id == GPUColumnTypeId::INT64)
      in.cols.i64_cols[i64_idx++] = reinterpret_cast<const int64_t*>(k->data_wrapper.data);
  }
  in.cols.n_ints   = i_idx;
  in.cols.n_int64s = i64_idx;

  out_payload_tbl.clear();
  // dst_slot is a RUNNING wire/result slot index (NOT the column index): each
  // numeric payload takes 1 slot, a VARCHAR takes VARCHAR_PAYLOAD_SLOTS. The
  // pack kernel iterates payload ENTRIES; the build/probe kernels copy SLOTS.
  int i64a_idx = 0, d_idx = 0, v_idx = 0, dst_slot = 0;
  for (int p = 0; p < static_cast<int>(payload.size()); ++p) {
    auto id      = payload[p]->data_wrapper.type.id();
    auto* data   = payload[p]->data_wrapper.data;
    if (id == GPUColumnTypeId::VARCHAR) {  // inline length-prefixed string → 4 slots
      in.cols.v_chars[v_idx] = reinterpret_cast<const uint8_t*>(data);
      in.cols.v_offsets[v_idx] =
          reinterpret_cast<const uint64_t*>(payload[p]->data_wrapper.offset);
      out_payload_tbl.push_back({JoinPayloadEntry::Src::VARCHAR, (int16_t)v_idx, (int16_t)dst_slot});
      ++v_idx;
      dst_slot += magi_generic::VARCHAR_PAYLOAD_SLOTS;
    } else if (id == GPUColumnTypeId::FLOAT64) {
      in.cols.d_cols[d_idx] = reinterpret_cast<const double*>(data);
      out_payload_tbl.push_back({JoinPayloadEntry::Src::DOUBLE, (int16_t)d_idx, (int16_t)dst_slot});
      ++d_idx; ++dst_slot;
    } else if (id == GPUColumnTypeId::INT32) {  // INTEGER — widened to int64 on the wire
      in.cols.i_cols[i_idx] = reinterpret_cast<const int32_t*>(data);
      out_payload_tbl.push_back({JoinPayloadEntry::Src::INT32, (int16_t)i_idx, (int16_t)dst_slot});
      ++i_idx; ++dst_slot;
    } else {  // INT64 / DECIMAL(≤18) stored as int64
      in.cols.i64_agg_cols[i64a_idx] = reinterpret_cast<const int64_t*>(data);
      out_payload_tbl.push_back({JoinPayloadEntry::Src::INT64, (int16_t)i64a_idx, (int16_t)dst_slot});
      ++i64a_idx; ++dst_slot;
    }
  }
  in.cols.n_ints       = i_idx;   // key INT32 cols + INT32 payload cols
  in.cols.n_doubles    = d_idx;
  in.cols.n_int64_aggs = i64a_idx;
  in.cols.n_varchars   = v_idx;

  in.cols.row_validity =
      (!keys.empty() && keys[0] && keys[0]->data_wrapper.validity_mask)
          ? reinterpret_cast<const uint32_t*>(keys[0]->data_wrapper.validity_mask)
          : nullptr;
  return in;
}

// ════════════════════════════════════════════════════════════════════════════
// Device-side AoS→SoA emit: build the output GPUColumns directly from the device
// JoinResultRow buffer (out_buf), with NO host round-trip. Replaces the
// D2H(slice) + CPU transpose + H2D path for join outputs (which is O(rows) on the
// host and dominates join→agg latency). Each kernel is a trivial data-parallel
// pass over the matched rows; VARCHAR needs a length pass + prefix sum + chars.
// ════════════════════════════════════════════════════════════════════════════
namespace {
__device__ __forceinline__ double jrr_slot(const magi_generic::JoinResultRow& r,
                                            int dst, bool from_build) {
  return from_build ? r.build_values[dst] : r.probe_values[dst];
}
__global__ void k_emit_key_i32(const magi_generic::JoinResultRow* in, size_t N,
                               int shift, int32_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (int32_t)((in[i].key_packed >> shift) & 0xFFFFFFFFull);
}
__global__ void k_emit_key_i64(const magi_generic::JoinResultRow* in, size_t N,
                               int shift, int64_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (int64_t)(in[i].key_packed >> shift);
}
__global__ void k_emit_f64(const magi_generic::JoinResultRow* in, size_t N,
                           int dst, bool fb, double* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = jrr_slot(in[i], dst, fb);
}
__global__ void k_emit_i32(const magi_generic::JoinResultRow* in, size_t N,
                           int dst, bool fb, int32_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { double dv = jrr_slot(in[i], dst, fb); int64_t x;
               __builtin_memcpy(&x, &dv, 8); out[i] = (int32_t)x; }
}
__global__ void k_emit_i64(const magi_generic::JoinResultRow* in, size_t N,
                           int dst, bool fb, int64_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { double dv = jrr_slot(in[i], dst, fb); int64_t x;
               __builtin_memcpy(&x, &dv, 8); out[i] = x; }
}
__global__ void k_emit_vc_len(const magi_generic::JoinResultRow* in, size_t N,
                              int dst, bool fb, uint64_t* len) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double dv = jrr_slot(in[i], dst, fb);
    unsigned char b8[8]; __builtin_memcpy(b8, &dv, 8);
    unsigned l = b8[0];
    if (l > magi_generic::VARCHAR_PAYLOAD_MAXLEN) l = magi_generic::VARCHAR_PAYLOAD_MAXLEN;
    len[i] = l;
  }
}
__global__ void k_emit_vc_chars(const magi_generic::JoinResultRow* in, size_t N,
                                int dst, bool fb, const uint64_t* offsets, uint8_t* chars) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    unsigned char buf[magi_generic::VARCHAR_PAYLOAD_SLOTS * 8];
    for (int s = 0; s < magi_generic::VARCHAR_PAYLOAD_SLOTS; ++s) {
      double dv = jrr_slot(in[i], dst + s, fb);
      __builtin_memcpy(buf + s * 8, &dv, 8);
    }
    unsigned l = buf[0];
    if (l > magi_generic::VARCHAR_PAYLOAD_MAXLEN) l = magi_generic::VARCHAR_PAYLOAD_MAXLEN;
    uint64_t o = offsets[i];
    for (unsigned b = 0; b < l; ++b) chars[o + b] = buf[1 + b];
  }
}
inline void emit_cfg(size_t N, int& tpb, unsigned& grid) {
  tpb = 256; grid = (unsigned)((N + tpb - 1) / tpb); if (grid == 0) grid = 1;
}
}  // namespace

shared_ptr<GPUColumn> EmitKeyColumnDevice(int gpu_id, const KeyFieldEntry& f,
                                          const magi_generic::JoinResultRow* d_in, size_t N,
                                          GPUBufferManager* gbm) {
  const int shift = f.byte_offset * 8;
  int tpb; unsigned grid; emit_cfg(N, tpb, grid);
  if (f.kind == KeyFieldKind::INT32) {
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr);
    auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
    k_emit_key_i32<<<grid, tpb>>>(d_in, N, shift, d);
    auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT32),
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  if (N == 0)
    return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
  auto* d = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
  k_emit_key_i64<<<grid, tpb>>>(d_in, N, shift, d);
  auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT64),
                                      reinterpret_cast<uint8_t*>(d), createNullMask(N));
  c->row_id_count = 0;
  return c;
}

shared_ptr<GPUColumn> EmitPayloadColumnDevice(int gpu_id, const shared_ptr<GPUColumn>& proto,
                                              const JoinPayloadEntry& e,
                                              const magi_generic::JoinResultRow* d_in, size_t N,
                                              bool from_build, GPUBufferManager* gbm) {
  int tpb; unsigned grid; emit_cfg(N, tpb, grid);
  if (e.src_kind == JoinPayloadEntry::Src::VARCHAR) {
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::VARCHAR),
                                        nullptr, nullptr, 0, true, nullptr);
    auto* d_len = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
    k_emit_vc_len<<<grid, tpb>>>(d_in, N, e.dst_idx, from_build, d_len);
    auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
    cudaMemset(d_offsets, 0, sizeof(uint64_t));
    thrust::inclusive_scan(thrust::device, d_len, d_len + N, d_offsets + 1);
    uint64_t total = 0;
    cudaMemcpy(&total, d_offsets + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto* d_chars = gbm->customCudaMalloc<uint8_t>(total == 0 ? 1 : total, gpu_id, false);
    k_emit_vc_chars<<<grid, tpb>>>(d_in, N, e.dst_idx, from_build, d_offsets, d_chars);
    auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::VARCHAR),
                                        d_chars, d_offsets, total, /*is_string_data=*/true,
                                        createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  if (e.src_kind == JoinPayloadEntry::Src::DOUBLE) {
    if (N == 0) return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
    auto* d = gbm->customCudaMalloc<double>(N, gpu_id, false);
    k_emit_f64<<<grid, tpb>>>(d_in, N, e.dst_idx, from_build, d);
    auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  if (e.src_kind == JoinPayloadEntry::Src::INT32) {
    if (N == 0) return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
    auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
    k_emit_i32<<<grid, tpb>>>(d_in, N, e.dst_idx, from_build, d);
    auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  // INT64 / DECIMAL — recover the int64 bit-pattern from the double slot.
  if (N == 0) return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
  auto* d = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
  k_emit_i64<<<grid, tpb>>>(d_in, N, e.dst_idx, from_build, d);
  auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,
                                      reinterpret_cast<uint8_t*>(d), createNullMask(N));
  c->row_id_count = 0;
  return c;
}

// ── Emit one key column from the result slice (inverse of pack_key_from_fields) ──
shared_ptr<GPUColumn> EmitKeyColumn(int gpu_id, const KeyFieldEntry& f,
                                    const std::vector<magi_generic::JoinResultRow>& slice,
                                    GPUBufferManager* gbm)
{
  const size_t N     = slice.size();
  const int    shift = f.byte_offset * 8;
  if (f.kind == KeyFieldKind::INT32) {
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr);
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i)
      v[i] = static_cast<int32_t>((slice[i].key_packed >> shift) & 0xFFFFFFFFu);
    auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
    cudaMemcpy(d, v.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT32),
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  // INT64
  if (N == 0)
    return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
  std::vector<int64_t> v(N);
  for (size_t i = 0; i < N; ++i)
    v[i] = static_cast<int64_t>(slice[i].key_packed >> shift);
  auto* d = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
  cudaMemcpy(d, v.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT64),
                                      reinterpret_cast<uint8_t*>(d), createNullMask(N));
  c->row_id_count = 0;
  return c;
}

// ── Emit one payload column from the slice ──────────────────────────────────
// Reads slice[i].probe_values[dst] (LHS) or build_values[dst] (RHS) per
// `from_build`. The slot holds raw 8 bytes (double bit-pattern for FLOAT64, the
// int64 bit-pattern bit-cast into a double for INT64/DECIMAL).
shared_ptr<GPUColumn> EmitPayloadColumn(int gpu_id, const shared_ptr<GPUColumn>& proto,
                                        const JoinPayloadEntry& e,
                                        const std::vector<magi_generic::JoinResultRow>& slice,
                                        bool from_build,
                                        GPUBufferManager* gbm)
{
  const size_t N  = slice.size();
  const auto   id = proto->data_wrapper.type.id();
  auto val = [&](size_t i) -> double {
    return from_build ? slice[i].build_values[e.dst_idx] : slice[i].probe_values[e.dst_idx];
  };
  if (e.src_kind == JoinPayloadEntry::Src::VARCHAR) {
    // Inline string: VARCHAR_PAYLOAD_SLOTS 8-byte slots from dst_idx hold
    // [byte0 = length][bytes 1.. = chars]. Rebuild a sirius VARCHAR column
    // (chars + uint64 offsets[N+1] + num_bytes), like the groupby's
    // EmitVarcharKeyFromPacked.
    auto read_buf = [&](size_t i, unsigned char* buf) {
      for (int s = 0; s < magi_generic::VARCHAR_PAYLOAD_SLOTS; ++s) {
        double dv = from_build ? slice[i].build_values[e.dst_idx + s]
                               : slice[i].probe_values[e.dst_idx + s];
        std::memcpy(buf + s * 8, &dv, sizeof(double));
      }
    };
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::VARCHAR),
                                        nullptr, nullptr, 0, true, nullptr);
    std::vector<uint64_t> h_offsets(N + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < N; ++i) {
      unsigned char buf[magi_generic::VARCHAR_PAYLOAD_SLOTS * 8];
      read_buf(i, buf);
      unsigned len = buf[0];
      if (len > magi_generic::VARCHAR_PAYLOAD_MAXLEN) len = magi_generic::VARCHAR_PAYLOAD_MAXLEN;
      total += len;
      h_offsets[i + 1] = total;
    }
    std::vector<uint8_t> h_chars(total == 0 ? 1 : total);
    for (size_t i = 0; i < N; ++i) {
      unsigned char buf[magi_generic::VARCHAR_PAYLOAD_SLOTS * 8];
      read_buf(i, buf);
      unsigned len = buf[0];
      if (len > magi_generic::VARCHAR_PAYLOAD_MAXLEN) len = magi_generic::VARCHAR_PAYLOAD_MAXLEN;
      size_t dst = h_offsets[i];
      for (unsigned b = 0; b < len; ++b) h_chars[dst++] = buf[1 + b];
    }
    auto* d_chars   = gbm->customCudaMalloc<uint8_t>(total == 0 ? 1 : total, gpu_id, false);
    auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
    if (total > 0) cudaMemcpy(d_chars, h_chars.data(), total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    auto c = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::VARCHAR),
                                        d_chars, d_offsets, total, /*is_string_data=*/true,
                                        createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  if (e.src_kind == JoinPayloadEntry::Src::DOUBLE) {
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
    std::vector<double> v(N);
    for (size_t i = 0; i < N; ++i) v[i] = val(i);
    auto* d = gbm->customCudaMalloc<double>(N, gpu_id, false);
    cudaMemcpy(d, v.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  if (e.src_kind == JoinPayloadEntry::Src::INT32) {
    // INTEGER — narrow the int64-widened value back to a 4-byte INT32 column
    // (preserves the type so a downstream join keyed on it still matches).
    if (N == 0)
      return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
    std::vector<int32_t> v(N);
    for (size_t i = 0; i < N; ++i) {
      int64_t x; double dv = val(i);
      std::memcpy(&x, &dv, sizeof(int64_t));
      v[i] = static_cast<int32_t>(x);
    }
    auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
    cudaMemcpy(d, v.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,
                                        reinterpret_cast<uint8_t*>(d), createNullMask(N));
    c->row_id_count = 0;
    return c;
  }
  // INT64 / DECIMAL — recover the int64 bit-pattern from the double slot.
  if (N == 0)
    return make_shared_ptr<GPUColumn>(0, proto->data_wrapper.type, nullptr, nullptr);
  std::vector<int64_t> v(N);
  for (size_t i = 0; i < N; ++i) {
    int64_t x; double dv = val(i);
    std::memcpy(&x, &dv, sizeof(int64_t));
    v[i] = x;
  }
  auto* d = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
  cudaMemcpy(d, v.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto c = make_shared_ptr<GPUColumn>(N, proto->data_wrapper.type,  // keep DECIMAL {w,scale}
                                      reinterpret_cast<uint8_t*>(d), createNullMask(N));
  c->row_id_count = 0;
  (void)id;
  return c;
}

}  // namespace

void Run(int                                       gpu_id,
         const std::vector<shared_ptr<GPUColumn>>& build_keys,
         const std::vector<shared_ptr<GPUColumn>>& build_payload,
         const std::vector<shared_ptr<GPUColumn>>& probe_keys,
         const std::vector<shared_ptr<GPUColumn>>& probe_payload,
         std::vector<shared_ptr<GPUColumn>>&       out_key,
         std::vector<shared_ptr<GPUColumn>>&       out_build_payload,
         std::vector<shared_ptr<GPUColumn>>&       out_payload)
{
  auto* gbm = &GPUBufferManager::GetInstance();

  // Build and probe must pack an IDENTICAL key — the single recipe below is
  // derived from probe_keys and reused to pack both sides, so equal logical
  // keys must produce equal bit patterns. Enforce matching arity + column types.
  if (build_keys.size() != probe_keys.size())
    throw NotImplementedException("magi_join: build/probe join-key arity mismatch");
  for (size_t i = 0; i < build_keys.size(); ++i)
    if (build_keys[i]->data_wrapper.type.id() != probe_keys[i]->data_wrapper.type.id())
      throw NotImplementedException("magi_join: build/probe join-key type mismatch (v1)");

  // One key recipe for both sides (equal keys must pack identically).
  magi_generic::KeyKind             key_kind;
  std::vector<KeyFieldEntry>        key_fields;
  if (!DeriveKeyShape(probe_keys, key_kind, key_fields))
    throw NotImplementedException("magi_join: unsupported join key shape (v1: fixed INT32/INT64)");

  // Build cardinality drives the tier (build keys, like groupby's PickTableSize).
  const uint64_t build_rows = build_keys.empty() ? 0 : build_keys[0]->column_length;
  const uint64_t est        = build_rows + build_rows / 3;
  magi_generic::TableSize ts =
      (est <= magi_generic::N_SLOTS_SMALL)  ? magi_generic::TableSize::SMALL
      : (est <= magi_generic::N_SLOTS_MEDIUM) ? magi_generic::TableSize::MEDIUM
      : (est <= magi_generic::N_SLOTS_LARGE)  ? magi_generic::TableSize::LARGE
                                              : magi_generic::TableSize::XLARGE;

  std::vector<JoinPayloadEntry> build_pl_tbl, probe_pl_tbl;
  auto build_in = BuildJoinInputs(build_keys, build_payload, build_pl_tbl);
  auto probe_in = BuildJoinInputs(probe_keys, probe_payload, probe_pl_tbl);

  // Output buffer from sirius's processing pool (NOT raw cudaMalloc → no OOM vs
  // the pre-reserved pool). Size ≈ this GPU's probe rows (≈ what the owner
  // receives, hash-partitioned) + 50% headroom for partition imbalance.
  const uint64_t probe_rows = probe_keys.empty() ? 0 : probe_keys[0]->column_length;
  const uint64_t out_cap    = std::max<uint64_t>(probe_rows + probe_rows / 2, 1);
  auto* out_buf = reinterpret_cast<magi_generic::JoinResultRow*>(
      gbm->customCudaMalloc<uint8_t>(out_cap * sizeof(magi_generic::JoinResultRow), gpu_id, false));
  auto* out_count_buf = gbm->customCudaMalloc<uint32_t>(1, gpu_id, false);

  static const bool DBG = std::getenv("MAGI_JOIN_DEBUG") != nullptr;
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<magi_generic::JoinResultRow> slice;  // unused under device_emit (API-required ref)
  const std::size_t emitted = magi_generic::distributed_hash_join_run_per_gpu(
      gpu_id, build_in, probe_in, key_fields,
      build_pl_tbl, probe_pl_tbl, key_kind, ts, out_buf, out_count_buf, out_cap, slice,
      /*device_emit=*/true);
  const auto t1 = std::chrono::high_resolution_clock::now();

  // Emit key + payload output columns ON-DEVICE, directly from the matched rows
  // still resident in `out_buf` (no D2H slice + host transpose round-trip).
  out_key.clear();
  for (auto& f : key_fields)
    out_key.push_back(EmitKeyColumnDevice(gpu_id, f, out_buf, emitted, gbm));
  out_payload.clear();
  for (int p = 0; p < static_cast<int>(probe_payload.size()); ++p)
    out_payload.push_back(EmitPayloadColumnDevice(
        gpu_id, probe_payload[p], probe_pl_tbl[p], out_buf, emitted, /*from_build=*/false, gbm));
  out_build_payload.clear();
  for (int p = 0; p < static_cast<int>(build_payload.size()); ++p)
    out_build_payload.push_back(EmitPayloadColumnDevice(
        gpu_id, build_payload[p], build_pl_tbl[p], out_buf, emitted, /*from_build=*/true, gbm));
  // The emit kernels + their thrust scans run on the default stream; make their
  // results visible before the columns are consumed downstream.
  cudaDeviceSynchronize();
  const auto t2 = std::chrono::high_resolution_clock::now();

  if (DBG) {
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[magi-join g%d] JOIN(build+probe)=%.1fms  DEVICE-EMIT(cols)=%.1fms  rows=%zu\n",
                 gpu_id, ms(t0, t1), ms(t1, t2), emitted);
  }
}

}  // namespace magi_join_op
}  // namespace duckdb
