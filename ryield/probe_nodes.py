"""One-off: list Aspen input node names of RYield R-101 and RStoic R-201 in case k."""
import os
import win32com.client as win32

SRC = r"D:\SAF_Nurul\res_20260403_non_CC\res_20260403_non_CC\260401_pyrolysis_oil_CC_case_k_6_non_CC.bkp"

aspen = win32.Dispatch("Apwn.Document")
aspen.InitFromArchive2(SRC)
aspen.Visible = False
aspen.SuppressDialogs = True
T = aspen.Tree
for b in ["R-101", "R-201"]:
    inp = T.FindNode(rf"\Data\Blocks\{b}\Input")
    print(f"== {b} Input:", [e.Name for e in inp.Elements])
    print(f"   {b} Ports:", [(p.Name, [s.Name for s in p.Elements]) for p in T.FindNode(rf"\Data\Blocks\{b}\Ports").Elements])
my = T.FindNode(r"\Data\Blocks\R-101\Input\MOLE_YIELD")
print("R-101 MOLE_YIELD children:", [e.Name for e in my.Elements][:5])
for k in ["BASIS_YIELD", "YIELD_BASIS", "BASIS", "OPT_YIELD", "SPEC_OPT", "TEMP", "PRES", "NPHASE"]:
    n = T.FindNode(rf"\Data\Blocks\R-101\Input\{k}")
    print(f"R-101 {k}:", None if n is None else (n.Value, [e.Name for e in n.Elements][:3] if n.Elements.Count else ""))
aspen.Close()
