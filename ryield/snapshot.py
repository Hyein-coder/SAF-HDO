"""Run an Aspen .bkp and dump stream component mass flows, T, P and block duties to JSON.

Usage: python snapshot.py <file.bkp> <out.json>
"""
import json
import sys
import win32com.client as win32

DUTY_NODES = ["B_TEMP", "B_PRES", "QCALC", "QNET", "HX_DUTY", "WNET", "BRAKE_POWER", "COND_DUTY", "REB_DUTY"]


def val(tree, path):
    n = tree.FindNode(path)
    return None if n is None else n.Value


def snapshot(aspen):
    T = aspen.Tree
    err = T.FindNode(r"\Data\Results Summary\Run-Status\Output\PER_ERROR")
    out = {"status": [e.Value for e in err.Elements], "streams": {}, "blocks": {}}
    for s in T.FindNode(r"\Data\Streams").Elements:
        o = rf"\Data\Streams\{s.Name}\Output"
        mf = T.FindNode(o + r"\MASSFLOW\MIXED")
        if mf is None:
            continue
        out["streams"][s.Name] = {
            "T": val(T, o + r"\TEMP_OUT\MIXED"),
            "P": val(T, o + r"\PRES_OUT\MIXED"),
            "mass": {c.Name: c.Value for c in mf.Elements},
        }
    for b in T.FindNode(r"\Data\Blocks").Elements:
        d = {k: val(T, rf"\Data\Blocks\{b.Name}\Output\{k}") for k in DUTY_NODES}
        out["blocks"][b.Name] = {k: v for k, v in d.items() if v is not None}
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    aspen = win32.Dispatch("Apwn.Document")
    try:
        aspen.InitFromArchive2(src)
        aspen.Visible = False
        aspen.SuppressDialogs = True
        aspen.Engine.Run2()
        snap = snapshot(aspen)
    finally:
        aspen.Close()
    json.dump(snap, open(dst, "w"), indent=1)
    print("status:", snap["status"], "| streams:", len(snap["streams"]), "| blocks:", len(snap["blocks"]))
