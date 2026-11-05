"""Swap HDO RStoic R-201/R-202/R-203 in case k for RYield with mass yields taken from the converged
RStoic run (base_k_rstoic.json from snapshot.py). Topology, T and P unchanged.

Usage: python convert_k.py <base.json> <src.bkp> <dst.bkp>
"""
import json
import re
import sys
import textwrap

# reactor: (inlet stream, outlet stream)
STAGES = {"R-201": ("207", "208"), "R-202": ("210", "211"), "R-203": ("212", "213")}


def stage_yields(snap, sin, sout):
    m_in, m_out = snap["streams"][sin]["mass"], snap["streams"][sout]["mass"]
    tot_in = sum(v or 0 for v in m_in.values())
    return {c: v / tot_in for c, v in m_out.items() if v and v > 0}, sum(v or 0 for v in m_out.values()) / tot_in


def cid(c):
    return c if re.fullmatch(r"[A-Z][A-Z0-9]*", c) else f'"{c}"'


def ryield_section(old, name, yields):
    """Rebuild one `? BLOCK RSTOIC "name" ?` section as RYIELD, keeping header options and the PRODUCTS/SS tail."""
    temp_pres = re.search(r"TEMP = \S+ <\d+> <\d+> PRES = \S+ <\d+> <\d+>", re.sub(r"\s+", " ", old)).group(0)
    tail = old[old.index("PRODUCTS"):]
    rows = " / ".join(f"CID = {cid(c)} SSID = MIXED MOLE-YIELD = {y:.12G} <0> <0> BASIS = MASS"
                      for c, y in yields.items())
    body = f"\\ PARAM {temp_pres} SPEC-OPT = TP \\ \\ MOLE-YIELD {rows} \\ \\ "
    head = old[:old.index("\\ PARAM")].replace("RSTOIC", "RYIELD").rstrip() + "\n"  # header may wrap lines
    return head + "\n".join(textwrap.wrap(body, 79, break_long_words=False, drop_whitespace=False)) + "\n" + tail


def main(base_json, src, dst):
    snap = json.load(open(base_json))
    t = open(src, encoding="latin-1").read()
    for r, (sin, sout) in STAGES.items():
        y, closure = stage_yields(snap, sin, sout)
        print(f"{r}: {len(y)} yields, sum(out)/sum(in) = {closure:.10f}")
        # flowsheet entry (may wrap across lines)
        t, n = re.subn(rf'(BLKID\s*=\s*"{r}"\s*BLKTYPE\s*=\s*)"RSTOIC"(\s*MDLTYPE\s*=\s*)"RStoic"', r'\1"RYIELD"\2"RYield"', t)
        assert n == 1, (r, "flowsheet", n)
        i = t.index(f'? BLOCK RSTOIC "{r}" ?')
        j = t.index("? BLOCK", i + 5)
        t = t[:i] + ryield_section(t[i:j], r, y) + t[j:]
    open(dst, "w", encoding="latin-1", newline="").write(t)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
