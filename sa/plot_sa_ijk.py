import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SADIR=r"D:\saf_hdo\sa"
CASES=['i','j','k']

# load tornado
torn={c:[] for c in CASES}
with open(os.path.join(SADIR,'sa_tornado_ijk.csv')) as f:
    for r in csv.DictReader(f):
        torn[r['case']].append((r['parameter'],float(r['base_msp']),float(r['low_msp']),float(r['high_msp']),float(r['swing'])))
# load oat
oat={c:{} for c in CASES}
with open(os.path.join(SADIR,'sa_oat_ijk.csv')) as f:
    for r in csv.DictReader(f):
        oat[r['case']].setdefault(r['parameter'],[]).append((float(r['value']),float(r['msp'])))

# ---- tornado: 3 panels ----
fig,axs=plt.subplots(1,3,figsize=(16,4.5),sharey=False)
for ax,c in zip(axs,CASES):
    rows=sorted(torn[c],key=lambda x:x[4])  # ascending swing -> biggest on top
    base=rows[0][1]
    names=[r[0] for r in rows]
    y=range(len(rows))
    for i,r in enumerate(rows):
        lo,hi=r[2],r[3]; left,right=min(lo,hi),max(lo,hi)
        ax.barh(i,right-left,left=left,color='#4C78A8',edgecolor='k',alpha=.85)
    ax.axvline(base,color='#E45756',lw=1.5,ls='--',label=f'base {base:.3f}')
    ax.set_yticks(list(y)); ax.set_yticklabels(names,fontsize=8)
    ax.set_title(f'Case {c}  (base MSP {base:.3f} $/kg)',fontsize=10)
    ax.set_xlabel('MSP [$/kg]'); ax.legend(fontsize=8,loc='lower right')
fig.suptitle('TEA sensitivity (catalyst OPEX in COM) — tornado, cases i/j/k',fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(SADIR,'tornado_ijk.png'),dpi=150); plt.close(fig)

# ---- spider: 3 panels, x=relative param change vs MSP ----
# normalize each param's grid to its base (index where value nearest base col) — use midpoint of grid as ref
fig,axs=plt.subplots(1,3,figsize=(16,4.5))
for ax,c in zip(axs,CASES):
    for name,pts in oat[c].items():
        pts=sorted(pts)
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        # x as % deviation from central grid value
        mid=xs[len(xs)//2]
        xrel=[100*(x-mid)/mid if mid else 0 for x in xs]
        ax.plot(xrel,ys,marker='o',ms=3,lw=1.2,label=name)
    ax.set_title(f'Case {c}',fontsize=10)
    ax.set_xlabel('parameter change from central value [%]'); ax.set_ylabel('MSP [$/kg]')
    ax.grid(alpha=.3)
axs[-1].legend(fontsize=7,loc='best')
fig.suptitle('TEA sensitivity (catalyst OPEX in COM) — spider, cases i/j/k',fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(SADIR,'spider_ijk.png'),dpi=150); plt.close(fig)
print("wrote tornado_ijk.png, spider_ijk.png")
