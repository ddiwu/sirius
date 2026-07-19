#!/usr/bin/env python3
"""Robust per-run result comparator for tools/qverify.sh.

Parses one or more duckdb CSV output files (headers off) that qverify.sh
produced, splits them into runs on the __QVERIFY_RUN_BOUNDARY__ marker rows,
and compares configs key-by-key with a float tolerance.

Hard-won rules encoded here (2026-07-05, Q3 debugging):
  - NEVER slice runs by physical line windows: [magi]/init banners are
    interleaved into stdout (45k+ junk lines seen), so windows drift.
    Data rows are selected by shape instead (majority field count + at
    least one float field + not a banner), runs split by explicit markers.
  - NEVER hash/md5 rows containing DOUBLE columns: cross-engine reduce
    order changes last bits, so equal results hash differently. Non-float
    fields form the row key; float fields are compared with a tolerance.
  - Rows with equal keys within one run are kept as a multiset.

Usage: qverify_compare.py [--rtol 1e-6] LABEL=FILE [LABEL=FILE ...]
The first LABEL=FILE is the reference (usually cpu=...). Exit 0 iff every
other config matches the reference and is internally consistent per run.
"""
import csv
import io as _io
import re
import sys

BOUNDARY = "__QVERIFY_RUN_BOUNDARY__"
INT_RE = re.compile(r"-?\d+")


def is_float_field(s):
    """True for numeric fields with a fractional/exponent part (measure
    columns). Pure integers (keys, counts) compare exactly."""
    if INT_RE.fullmatch(s):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def split_csv(line):
    """CSV-aware split: honors quoted fields (TPC-H comment/address columns
    contain commas; a naive split misclassified those rows as junk and the
    compare silently ran on the comma-free subset only)."""
    if '"' not in line:
        return line.split(",")
    try:
        return next(csv.reader(_io.StringIO(line)))
    except (csv.Error, StopIteration):
        return line.split(",")


def parse_runs(path):
    """-> (runs, n_junk). Each run: {key_tuple: sorted list of value tuples}."""
    raw_runs, cur, junk = [], [], 0
    lines = []
    for raw in open(path, errors="replace"):
        line = raw.strip().replace("\r", "")
        if line:
            lines.append(line)
    # Majority field count over plausible data lines picks the result shape.
    counts = {}
    for line in lines:
        if BOUNDARY in line or line.startswith(("[", "=")):
            continue
        f = split_csv(line)
        if any(is_number(x) for x in f):
            counts[len(f)] = counts.get(len(f), 0) + 1
    shape = max(counts, key=counts.get) if counts else -1
    for line in lines:
        if BOUNDARY in line:
            if cur:
                raw_runs.append(cur)
            cur = []
            continue
        f = split_csv(line)
        if (len(f) == shape and not line.startswith(("[", "=")) and
                any(is_number(x) for x in f)):
            cur.append(f)
        else:
            junk += 1
    if cur:
        raw_runs.append(cur)
    runs, ordered = [], []
    for rows in raw_runs:
        groups = {}
        seq = []  # (key_tuple, val_tuple) in output order — for --ordered
        for f in rows:
            key = tuple(x for x in f if not is_float_field(x))
            val = tuple(float(x) for x in f if is_float_field(x))
            groups.setdefault(key, []).append(val)
            seq.append((key, val))
        for k in groups:
            groups[k].sort()
        runs.append(groups)
        ordered.append(seq)
    return runs, junk, ordered


def _rows_equal(ra, rb, rtol):
    """Row equal iff non-float key columns match exactly and float columns
    match within rtol. Returns (equal, max_rel_seen)."""
    (ka, va), (kb, vb) = ra, rb
    if ka != kb or len(va) != len(vb):
        return False, 0.0
    mr = 0.0
    for x, y in zip(va, vb):
        rel = abs(x - y) / max(abs(y), 1e-30)
        mr = max(mr, rel)
        if rel > rtol:
            return False, mr
    return True, mr


def diff_ordered(a, b, rtol, tie_window=16):
    """Positional row-by-row compare (order-sensitive, for ORDER BY).
    Rows may permute WITHIN a near-tie group: when the sort key is a float
    aggregate, rows whose keys differ only at ~1e-16 can order arbitrarily
    between engines (different reduce order) — any such permutation is a valid
    ORDER BY result, and with many ties it is NOT limited to adjacent swaps
    (Q11-order showed 3-cycles). A mismatch at position i is soft iff a[i]
    matches some unconsumed b[j] within tie_window positions; anything else is
    a hard diff. -> (n_hard_diff, n_soft_swap, max_rel)."""
    n_hard, n_soft, max_rel = 0, 0, 0.0
    if len(a) != len(b):
        n_hard += abs(len(a) - len(b))
    L = min(len(a), len(b))
    consumed = set()  # b-indices already matched (each b row usable once)
    for i in range(L):
        if i not in consumed:
            eq, mr = _rows_equal(a[i], b[i], rtol)
            max_rel = max(max_rel, mr)
            if eq:
                consumed.add(i)
                continue
        found = False
        for j in range(max(0, i - tie_window), min(L, i + tie_window + 1)):
            if j in consumed:
                continue
            ej, mj = _rows_equal(a[i], b[j], rtol)
            if ej:
                max_rel = max(max_rel, mj)
                consumed.add(j)
                n_soft += 1
                found = True
                break
        if not found:
            n_hard += 1
    return n_hard, n_soft, max_rel


def diff(a, b, rtol):
    """-> (n_key_diff, n_val_diff, max_rel). Compare two runs."""
    only_a, only_b = a.keys() - b.keys(), b.keys() - a.keys()
    n_val, max_rel = 0, 0.0
    for k in a.keys() & b.keys():
        va, vb = a[k], b[k]
        if len(va) != len(vb):
            n_val += 1
            continue
        for ta, tb in zip(va, vb):
            for x, y in zip(ta, tb):
                rel = abs(x - y) / max(abs(y), 1e-30)
                max_rel = max(max_rel, rel)
                if rel > rtol:
                    n_val += 1
    return len(only_a) + len(only_b), n_val, max_rel


def checksum(run):
    return sum(x for vals in run.values() for t in vals for x in t)


def main():
    args = sys.argv[1:]
    rtol = 1e-6
    ordered = False
    while args and args[0] in ("--rtol", "--ordered"):
        if args[0] == "--rtol":
            rtol = float(args[1])
            args = args[2:]
        else:  # --ordered: compare rows positionally (ORDER BY correctness)
            ordered = True
            args = args[1:]
    configs = []
    for a in args:
        label, _, path = a.partition("=")
        runs, junk, ordruns = parse_runs(path)
        configs.append((label, runs, junk, ordruns))
        sizes = [len(r) for r in runs]
        sums = " ".join(f"{checksum(r):.4f}" for r in runs[:4])
        print(f"[{label}] runs={len(runs)} rows/run={sizes} junk_lines={junk}"
              f"{' [ordered]' if ordered else ''}")
        print(f"[{label}] checksums: {sums}")
    if not configs:
        print("usage: qverify_compare.py [--rtol X] [--ordered] ref=FILE [cfg=FILE ...]")
        return 2
    ok = True
    ref_label, ref_runs, _, ref_ord = configs[0]
    if not ref_runs:
        print(f"FAIL: reference config '{ref_label}' produced no runs")
        return 1
    ref = ref_runs[0]
    if ordered:
        # Order-sensitive: row-by-row against the reference's first run.
        for label, _, _, ordruns in configs:
            if not ordruns:
                print(f"FAIL [{label}]: no data runs parsed")
                ok = False
                continue
            for i, r in enumerate(ordruns):
                base = ordruns[0] if label == ref_label else ref_ord[0]
                if label == ref_label and i == 0:
                    continue
                hard, soft, mr = diff_ordered(r, base, rtol)
                tag = f"run{i} vs run0" if label == ref_label else f"vs [{ref_label}]"
                status = "OK  " if hard == 0 else "FAIL"
                swap = f", near_tie_swaps={soft}" if soft else ""
                print(f"{status} [{label}] {tag}: rows {len(r)} vs {len(base)}, "
                      f"hard_diffs={hard}{swap} max_rel={mr:.3e}")
                if hard:
                    ok = False
        print("VERDICT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    for label, runs, _, _ in configs:
        if not runs:
            print(f"FAIL [{label}]: no data runs parsed")
            ok = False
            continue
        for i, r in enumerate(runs[1:], 1):  # internal consistency
            kd, vd, mr = diff(runs[0], r, rtol)
            if kd or vd:
                print(f"FAIL [{label}] run0 vs run{i}: key_diffs={kd} "
                      f"value_diffs={vd} max_rel={mr:.3e}")
                ok = False
        if label != ref_label:
            kd, vd, mr = diff(runs[0], ref, rtol)
            status = "OK  " if not (kd or vd) else "FAIL"
            print(f"{status} [{label}] vs [{ref_label}]: rows {len(runs[0])} vs "
                  f"{len(ref)}, key_diffs={kd} value_diffs={vd} max_rel={mr:.3e}")
            if kd or vd:
                ok = False
                # show a few offenders to start debugging from
                shown = 0
                for k in list(runs[0].keys() - ref.keys())[:3]:
                    print(f"      only in {label}: {k}")
                for k in list(ref.keys() - runs[0].keys())[:3]:
                    print(f"      only in {ref_label}: {k}")
                for k in runs[0].keys() & ref.keys():
                    if runs[0][k] != ref[k] and shown < 3:
                        kd2, vd2, mr2 = diff({k: runs[0][k]}, {k: ref[k]}, rtol)
                        if vd2:
                            print(f"      value diff at {k}: "
                                  f"{len(runs[0][k])}x{runs[0][k][:1]} vs "
                                  f"{len(ref[k])}x{ref[k][:1]}")
                            shown += 1
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
