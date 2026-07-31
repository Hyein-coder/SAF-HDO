import importlib.util,sys,csv,os
_here=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("m",os.path.join(_here,"sa_model_ijk.py"))
m=importlib.util.module_from_spec(spec); sys.argv=['x']; spec.loader.exec_module(m)
cases=m.cases; solve=m.solve_msp

# parameter grids (from sa/SA_driver.m)
def sweeps():
    return [
      ('Hydrogen price ($/kg)','H2',      [1.00,1.50,2.00,2.50,3.34], lambda v:{}),  # inert: no H2 term
      ('Feedstock cost (x)',   'bio_mult',[0.50,0.75,1.00,1.25,1.50], lambda v:{'bio_mult':v}),
      ('Catalyst lifetime (yr)','cat_year',[1,2,3,4,5],               lambda v:{'cat_year':v,'cat_replace':True}),
      ('Discount rate (-)',    'irate',   [0.08,0.10,0.12,0.15,0.18,0.20], lambda v:{'irate':v}),
      ('Plant life (yr)',      'plife',   [10,12,15,18,20],           lambda v:{'plife':v}),
      ('CAPEX multiplier (-)', 'capex',   [0.70,0.85,1.00,1.15,1.30], lambda v:{'capex_mult':v}),
      ('Natural gas price (x)','ng_mult', [0.50,0.75,1.00,1.25,1.50], lambda v:{'ng_mult':v}),
    ]

SADIR=r"D:\saf_hdo\sa"
# base MSP per case WITH catalyst replacement (this TEA version); and published (no replace)
base={}; pub={}
for c,d in cases.items():
    base[c]=solve(d,cat_replace=True,cat_year=3)
    pub[c]=solve(d,cat_replace=False)

# OAT csv
with open(os.path.join(SADIR,'sa_oat_ijk.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['case','parameter','value','msp'])
    for c,d in cases.items():
        for name,key,grid,mk in sweeps():
            for v in grid:
                kw=dict(cat_replace=True,cat_year=3); kw.update(mk(v))
                w.writerow([c,name,v,round(solve(d,**kw),6)])

# tornado csv + print
rows=[]
with open(os.path.join(SADIR,'sa_tornado_ijk.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['case','parameter','base_msp','low_value','low_msp','high_value','high_msp','swing'])
    for c,d in cases.items():
        for name,key,grid,mk in sweeps():
            lo,hi=grid[0],grid[-1]
            kwlo=dict(cat_replace=True,cat_year=3); kwlo.update(mk(lo))
            kwhi=dict(cat_replace=True,cat_year=3); kwhi.update(mk(hi))
            mlo=solve(d,**kwlo); mhi=solve(d,**kwhi)
            swing=abs(mhi-mlo)
            w.writerow([c,name,round(base[c],6),lo,round(mlo,6),hi,round(mhi,6),round(swing,6)])
            rows.append((c,name,base[c],lo,mlo,hi,mhi,swing))

print("Base MSP ($/kg): published (no catalyst OPEX) vs this-TEA (catalyst replacement ON):")
for c in cases: print(f"  {c}: published {pub[c]:.5f}  ->  +catalyst {base[c]:.5f}  (delta {base[c]-pub[c]:+.5f})")

for c in cases:
    print(f"\n=== case {c}  (base MSP {base[c]:.4f} $/kg) — tornado, sorted by swing ===")
    rr=[r for r in rows if r[0]==c]; rr.sort(key=lambda r:-r[7])
    print(f"  {'parameter':24}{'low->MSP':>20}{'high->MSP':>20}{'swing $/kg':>12}")
    for c_,name,b,lo,mlo,hi,mhi,sw in rr:
        print(f"  {name:24}{f'{lo}->{mlo:.4f}':>20}{f'{hi}->{mhi:.4f}':>20}{sw:12.5f}")
print("\nWrote sa_oat_ijk.csv, sa_tornado_ijk.csv to",SADIR)
