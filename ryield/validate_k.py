"""Case k RStoic vs RYield: key stream flows, phase fractions, and experimental-validation metrics.

Metrics reuse aspen_utils.AspenSim and the criteria of drHa_rxn_tuning.py (H/C, O/C of HDO oil 302,
liquid and oil yield on HDO effluent 216, carbon-number distribution of 216 vs HDO_exp.xlsx column k).

Usage: python validate_k.py run <file.bkp> <out.json>      (one Aspen file per process: terminate() uninitializes COM)
       python validate_k.py report <rstoic.json> <ryield.json>
"""
import json
import os
import sys

STREAMS = {
    "103": "biomass feed (wet)",
    "201": "fresh H2 feed",
    "205": "H2 to HDO (fresh + recycle)",
    "146": "pyrolysis oil to HDO",
    "207": "HDO feed (R-201 in)",
    "208": "R-201 out",
    "211": "R-202 out",
    "213": "R-203 out",
    "216": "HDO effluent (cooled)",
    "301": "aqueous phase",
    "302": "HDO oil",
    "326": "SAF",
    "324": "gasoline",
    "321": "heavy/diesel-range (burned in R-301)",
    "107": "flue gas S-101",
    "142": "flue gas S-104",
    "330": "CO2 S-302",
    "424": "flue gas R-403",
}
CO2_STREAMS = ["107", "142", "330", "424"]
TARGET = {"k": {"H/C": 1.8809, "O/C": 0.0072}, "bench": {"H/C": 1.9856, "O/C": 0.0040, "liquid": 0.841, "oil": 0.421}}


def run(bkp, out):
    here = os.path.dirname(os.path.abspath(__file__))
    bkp, out = os.path.abspath(bkp), os.path.abspath(out)
    sys.path.insert(0, os.path.dirname(here))
    os.chdir(r"D:\saf_hdo")  # aspen_utils loads HDO_exp.xlsx and aspen\251009_..._normalized.xlsx relative to cwd
    from aspen_utils import AspenSim

    sim = AspenSim(bkp, case_target="k", visible=False, rxtor_nodes=[])
    T = sim.aspen.Tree

    def v(path):
        n = T.FindNode(path)
        return None if n is None else n.Value

    res = {"status": sim.check_result_status(), "streams": {}}
    for s in STREAMS:
        o = rf"\Data\Streams\{s}\Output"
        res["streams"][s] = {
            "total": v(o + r"\MASSFLMX\$TOTAL"),
            "mixed": v(o + r"\MASSFLMX\MIXED"),
            "biomass_nc": v(o + r"\MASSFLOW\NC\BIOMASS"),
            "H2": v(o + r"\MASSFLOW\MIXED\H2"),
            "CO2": v(o + r"\MASSFLOW\MIXED\CO2"),
            "water": v(o + r"\MASSFLOW\MIXED\WATER"),
            "VFRAC": v(o + r"\STR_MAIN\VFRAC\MIXED"),
            "LFRAC": v(o + r"\STR_MAIN\LFRAC\MIXED"),
            "T": v(o + r"\TEMP_OUT\MIXED"),
            "P": v(o + r"\PRES_OUT\MIXED"),
        }
    el = sim.get_elemental_composition("302")
    product, oil, water = (sim.get_mass_flow(s) for s in ("216", "302", "301"))
    res["exp"] = {
        "H/C": el["H"] / (el["C"] / 12),
        "O/C": (el["O"] / 16) / (el["C"] / 12),
        "liquid": (oil + water) / product,
        "oil": oil / product,
        "aqueous": water / product,
        "oil_water_frac": sim.get_frac_water("302"),
        "oil_elem": el,
    }
    pel, pw = sim.get_elemental_composition_wo_water(sim.pyro_prod_stream)
    res["pyro"] = {"elem_dry": pel, "water_frac": pw}
    cc = sim.get_carbon_number_composition(sim.prod_stream)
    res["cc"] = {str(k): x for k, x in cc.items()}
    res["cc_target"] = {str(k): x for k, x in sim.target.items()}
    sim.terminate()
    json.dump(res, open(out, "w"), indent=1)
    print("status:", res["status"], "->", out)


def fmt(x, nd=2):
    return "-" if x is None else f"{x:,.{nd}f}"


def rel(a, b):
    return "-" if a in (None, 0) or b is None else f"{(b - a) / abs(a):+.1e}"


def report(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    print(f"status: RStoic={a['status']}  RYield={b['status']}\n")
    print("## 1. Mass flows (kg/h)\n")
    print("| Stream | Role | Quantity | RStoic | RYield | Rel. diff |\n|---|---|---|---|---|---|")
    for s, role in STREAMS.items():
        sa, sb = a["streams"][s], b["streams"][s]
        rows = [("total", "total")]
        if s == "103":
            rows.append(("biomass_nc", "dry biomass (NC)"))
        if s in ("201", "205", "207", "208", "211", "213", "216"):
            rows.append(("H2", "H2"))
        if s in CO2_STREAMS:
            rows.append(("CO2", "CO2"))
        for k, label in rows:
            print(f"| {s} | {role} | {label} | {fmt(sa[k])} | {fmt(sb[k])} | {rel(sa[k], sb[k])} |")
    co2a = sum(a["streams"][s]["CO2"] or 0 for s in CO2_STREAMS)
    co2b = sum(b["streams"][s]["CO2"] or 0 for s in CO2_STREAMS)
    print(f"| {'+'.join(CO2_STREAMS)} | total CO2 emitted | CO2 | {fmt(co2a)} | {fmt(co2b)} | {rel(co2a, co2b)} |")
    h2a = a["streams"]["207"]["H2"] - a["streams"]["216"]["H2"]
    h2b = b["streams"]["207"]["H2"] - b["streams"]["216"]["H2"]
    print(f"| 207-216 | H2 consumed in HDO | H2 | {fmt(h2a)} | {fmt(h2b)} | {rel(h2a, h2b)} |")

    print("\n## 2. Phase fractions (molar)\n")
    print("| Stream | Role | T (°C) | P (bar) | VFRAC RStoic | VFRAC RYield | LFRAC RStoic | LFRAC RYield |\n|---|---|---|---|---|---|---|---|")
    for s, role in STREAMS.items():
        sa, sb = a["streams"][s], b["streams"][s]
        print(f"| {s} | {role} | {fmt(sa['T'], 1)} | {fmt(sa['P'], 1)} | {fmt(sa['VFRAC'], 5)} | {fmt(sb['VFRAC'], 5)} "
              f"| {fmt(sa['LFRAC'], 5)} | {fmt(sb['LFRAC'], 5)} |")

    print("\n## 3. Experimental validation (criteria from drHa_rxn_tuning.py)\n")
    print("| Metric | Exp. case k | Exp. bench | RStoic | RYield |\n|---|---|---|---|---|")
    for k in ("H/C", "O/C", "liquid", "oil", "aqueous", "oil_water_frac"):
        print(f"| {k} | {fmt(TARGET['k'].get(k), 4)} | {fmt(TARGET['bench'].get(k), 4)} | {fmt(a['exp'][k], 4)} | {fmt(b['exp'][k], 4)} |")
    for e in "CHON":
        print(f"| oil 302 mass frac {e} | - | - | {fmt(a['exp']['oil_elem'][e], 4)} | {fmt(b['exp']['oil_elem'][e], 4)} |")
    for e in "CHON":
        print(f"| pyro oil 146 dry mass frac {e} | - | - | {fmt(a['pyro']['elem_dry'][e], 4)} | {fmt(b['pyro']['elem_dry'][e], 4)} |")
    print(f"| pyro oil 146 water frac | - | - | {fmt(a['pyro']['water_frac'], 4)} | {fmt(b['pyro']['water_frac'], 4)} |")

    print("\nCarbon-number distribution of 216 (hydrocarbon mass fraction) vs HDO_exp.xlsx column k:\n")
    print("| C# | Exp. k | RStoic | RYield |\n|---|---|---|---|")
    t = a["cc_target"]
    for n in t:
        print(f"| {n} | {t[n]:.4f} | {a['cc'][n]:.4f} | {b['cc'][n]:.4f} |")
    for lab, r in (("RStoic", a), ("RYield", b)):
        err = [r["cc"][n] - t[n] for n in t]
        print(f"{lab}: MAE = {sum(map(abs, err)) / len(err):.5f}, MSE = {sum(e * e for e in err) / len(err):.3e}")
    d = max(abs(a["cc"][n] - b["cc"][n]) for n in t)
    print(f"max |RStoic - RYield| over C#: {d:.2e}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]](*sys.argv[2:4])
