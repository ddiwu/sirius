// Fused join->aggregate descriptor + device mini-VM.
//
// When the magi shuffle-join feeds an ungrouped aggregate over a projection of
// the join output (e.g. TPC-H Q14: 100*sum(case when p_type like 'PROMO%' then
// l_extendedprice*(1-l_discount) else 0)/sum(l_extendedprice*(1-l_discount))),
// the probe kernel can evaluate the aggregate-input expression(s) per match and
// accumulate directly into a per-GPU accumulator — overlapping the agg with the
// NVLink shuffle and skipping the out_buf materialization + a separate
// projection + aggregate pass. This header carries the compact opcode "program"
// (built host-side from the BoundExpression tree) and the device stack-VM that
// the probe kernel runs per match.
#pragma once
#include <cstdint>

namespace duckdb {
namespace magi_fused {

// Stack-VM opcodes. Operands push/pop doubles. LOAD_PROBE/LOAD_BUILD read the
// matched row's probe_values[arg] / build_values[arg] (the JoinResultRow slots);
// LIKE_PREFIX tests an inlined-VARCHAR build slot against a prefix pattern.
enum VMOp : uint8_t {
  OP_HALT = 0,
  OP_LOAD_PROBE,   // push probe_values[arg]
  OP_LOAD_BUILD,   // push build_values[arg]
  OP_LOAD_CONST,   // push consts[arg]
  OP_ADD,          // b=pop,a=pop, push a+b
  OP_SUB,          // push a-b
  OP_MUL,          // push a*b
  OP_DIV,          // push a/b
  OP_LIKE_PREFIX,  // push (build inlined-varchar at patterns[arg].build_slot starts-with pat) ? 1:0
  OP_SELECT,       // els=pop,then=pop,cond=pop, push cond!=0 ? then : els  (CASE WHEN)
  // DECIMAL/INT64 payload slots hold the raw int64 bit-pattern (bit-cast into the
  // double slot); these loads recover it and divide by 10^scale (dec_div) to a double.
  OP_LOAD_PROBE_DEC,  // push (int64-bits of probe_values[arg]) / dec_div
  OP_LOAD_BUILD_DEC,  // push (int64-bits of build_values[arg]) / dec_div
};

struct VMInstr {
  uint8_t op;
  uint8_t arg;
};

enum AggKind : uint8_t { AGG_SUM = 0, AGG_COUNT = 1 };

static constexpr int MAX_FUSED_AGGS = 4;
static constexpr int MAX_INSTRS     = 64;
static constexpr int MAX_CONSTS     = 16;
static constexpr int MAX_PATTERNS   = 4;
static constexpr int PATTERN_MAXLEN = 16;
static constexpr int VM_STACK       = 16;

struct LikePattern {
  int  build_slot;             // base slot of the inlined VARCHAR in build_values[]
  int  plen;                   // prefix length to compare
  char pat[PATTERN_MAXLEN];
};

// A flat program covering up to MAX_FUSED_AGGS accumulators. Agg `a`'s opcodes are
// instrs[prog_off[a] .. prog_off[a+1]); its result is summed (or counted) into the
// per-GPU accumulator slot `a`.
struct FusedAggProgram {
  int         n_aggs;
  int         prog_off[MAX_FUSED_AGGS + 1];
  uint8_t     kind[MAX_FUSED_AGGS];
  VMInstr     instrs[MAX_INSTRS];
  double      consts[MAX_CONSTS];
  LikePattern patterns[MAX_PATTERNS];
  int         n_patterns;
  double      dec_div;   // 10^scale for OP_LOAD_*_DEC (1.0 if unused)
};

#ifdef __CUDACC__
// Evaluate aggregate `a`'s opcode program over one match's value arrays.
__device__ __forceinline__ double
magi_vm_eval(const FusedAggProgram& p, int a,
             const double* __restrict__ build_values,
             const double* __restrict__ probe_values) {
  double st[VM_STACK];
  int    sp = 0;
  const int end = p.prog_off[a + 1];
  for (int pc = p.prog_off[a]; pc < end; ++pc) {
    const VMInstr ins = p.instrs[pc];
    switch (ins.op) {
      case OP_LOAD_PROBE: st[sp++] = probe_values[ins.arg]; break;
      case OP_LOAD_BUILD: st[sp++] = build_values[ins.arg]; break;
      case OP_LOAD_CONST: st[sp++] = p.consts[ins.arg]; break;
      case OP_ADD: { double b = st[--sp], aa = st[--sp]; st[sp++] = aa + b; } break;
      case OP_SUB: { double b = st[--sp], aa = st[--sp]; st[sp++] = aa - b; } break;
      case OP_MUL: { double b = st[--sp], aa = st[--sp]; st[sp++] = aa * b; } break;
      case OP_DIV: { double b = st[--sp], aa = st[--sp]; st[sp++] = aa / b; } break;
      case OP_LIKE_PREFIX: {
        const LikePattern& lp = p.patterns[ins.arg];
        unsigned char buf[32];
        for (int s = 0; s < 4; ++s) {
          double dv = build_values[lp.build_slot + s];
          __builtin_memcpy(buf + s * 8, &dv, 8);
        }
        int len = buf[0];
        int ok  = (len >= lp.plen);
        for (int b = 0; ok && b < lp.plen; ++b)
          if ((char)buf[1 + b] != lp.pat[b]) ok = 0;
        st[sp++] = ok ? 1.0 : 0.0;
      } break;
      case OP_SELECT: {
        double els = st[--sp], then = st[--sp], cond = st[--sp];
        st[sp++] = (cond != 0.0) ? then : els;
      } break;
      case OP_LOAD_PROBE_DEC: {
        long long raw; double dv = probe_values[ins.arg];
        __builtin_memcpy(&raw, &dv, 8); st[sp++] = (double)raw / p.dec_div;
      } break;
      case OP_LOAD_BUILD_DEC: {
        long long raw; double dv = build_values[ins.arg];
        __builtin_memcpy(&raw, &dv, 8); st[sp++] = (double)raw / p.dec_div;
      } break;
      case OP_HALT: pc = end; break;
    }
  }
  return (sp > 0) ? st[sp - 1] : 0.0;
}
#endif  // __CUDACC__

// ── Side channel ────────────────────────────────────────────────────────────
// The fused join writes its per-GPU partial aggregate sums here; the result
// collector (host) reduces them across GPUs and applies the query's top
// projection (e.g. Q14's 100*a/b), bypassing the materialized join output +
// the projection + aggregate passes. Single-query lifetime; the collector resets
// `active` after consuming.
static constexpr int FUSED_MAX_GPUS = 8;
struct FusedResultChannel {
  bool   active;
  int    n_aggs;
  double partials[FUSED_MAX_GPUS][MAX_FUSED_AGGS];
};
FusedResultChannel& fused_channel();  // singleton, defined in magi_join_runtime.cu

}  // namespace magi_fused
}  // namespace duckdb
