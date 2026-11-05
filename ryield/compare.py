"""Diff two snapshot.py JSONs. Flags values with |diff| > ABS and |diff|/|base| > REL.

Usage: python compare.py <base.json> <new.json>
"""
import json
import sys

REL, ABS = 1e-4, 1e-6


def bad(a, b):
    a, b = a or 0.0, b or 0.0
    d = abs(b - a)
    return d > ABS and d > REL * max(abs(a), abs(b)), d


base, new = (json.load(open(p)) for p in sys.argv[1:3])
print("status base:", base["status"], "\nstatus new :", new["status"])
flags, worst = [], (0.0, "")
for s, rec in base["streams"].items():
    nrec = new["streams"].get(s)
    if nrec is None:
        flags.append(f"stream {s} missing")
        continue
    pairs = [(f"{s} {k}", rec[k], nrec[k]) for k in ("T", "P")]
    pairs += [(f"{s} mass[{c}]", v, nrec["mass"].get(c)) for c, v in rec["mass"].items()]
    for label, a, b in pairs:
        f, d = bad(a, b)
        rel = d / max(abs(a or 0), abs(b or 0), 1e-30)
        if rel > worst[0] and d > ABS:
            worst = (rel, label)
        if f:
            flags.append(f"{label}: {a} -> {b}")
for blk, rec in base["blocks"].items():
    for k, a in rec.items():
        b = new["blocks"].get(blk, {}).get(k)
        if bad(a, b)[0]:
            flags.append(f"block {blk} {k}: {a} -> {b}")
print(f"worst stream rel diff: {worst[0]:.3e} at {worst[1]}")
print(f"{len(flags)} flagged (REL>{REL}, ABS>{ABS}):")
print("\n".join(flags[:60]))
