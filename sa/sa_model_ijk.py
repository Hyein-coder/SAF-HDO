"""Faithful Python port of sa/cash_flow_SA.m, driven by frozen Excel primitives
for cases i, j, k (non_CC). Reproduces published MSP, then runs the reviewer SA
levers using the catalyst-OPEX-in-COM TEA (catalyst replacement charged)."""
import numpy as np

# ---- fixed COST (from TEA_LCA_non_CC_260407 COST_PARAMETER.m + matlabBC utility) ----
TAX=0.3892; IRATE0=0.15; PLIFE0=15; CONSTR=2; TOP=8000
P_GAS=1.22
MACRS7=np.array([0.1429,0.2449,0.1749,0.1249,0.0893,0.0892,0.0893,0.0446])
C_CAT_INV=59707.35
# labor (verified: sums to Excel Operation 3,661,867)
C_O1=35*10*TOP; C_O2=0.15*C_O1; C_O3=0.06*C_O2; C_O4=60000*10/3; C_O5=65000*10/3
C_O_sum=C_O1+C_O2+C_O3+C_O4+C_O5
# base feedstock prices implied by Excel ($/kg); WF same all cases
BIO_PRICE0=0.06923*(0.054673/0.0363)   # escalated; ~0.10427
NG_PRICE0=0.054673
WF_MASS=34442096.91430799/(TOP*BIO_PRICE0)

# ---- per-case frozen primitives (Excel TEA sheet) ----
cases={
 'j':dict(mSAF=7260.40883,mGAS=1046.90593,mNG=3604.44039,
          STEAM=702731.8703345298,ELEC=1405448.0663827967,REF=83822.54088302005,WWT=34173.005064,
          EQ=25384301.329575393, MSP_pub=1.0846549416834341),
 'k':dict(mSAF=8307.07591,mGAS=1110.03708,mNG=3982.17033,
          STEAM=707062.3449515969,ELEC=1456057.8743899968,REF=88000.94504016126,WWT=34933.148184,
          EQ=25800788.345711607, MSP_pub=0.94957253257358),
 'i':dict(mSAF=5994.0714,mGAS=635.629964,mNG=3987.91247,
          STEAM=640185.6252688643,ELEC=1456041.4777666365,REF=80148.85250047174,WWT=35069.031888,
          EQ=25476761.16708133, MSP_pub=1.4061299773624727),
}

def npv(x, d, *, capex_mult=1.0, cat_replace=True, cat_year=3, C_cat_inv=C_CAT_INV,
        bio_mult=1.0, ng_mult=1.0, irate=IRATE0, plife=PLIFE0):
    S = d['mSAF']*TOP*x + d['mGAS']*TOP*P_GAS
    # CAPEX buildup (matches cash_flow_SA)
    C_EQP=(d['EQ']-C_CAT_INV)*capex_mult
    C_TBM=C_EQP+20000+C_cat_inv
    C_site=0.1*C_TBM
    sum_alloc=  d['sum_alloc']
    C_alloc=0.05*C_TBM+sum_alloc
    C_DPI=C_TBM+C_site+0+C_alloc
    C_cont=0.15*C_DPI
    C_TDC=C_DPI+C_cont
    # operating
    C_feed=np.array([34442096.91430799*bio_mult, d['mNG']*TOP*NG_PRICE0*ng_mult])
    C_util=np.array([d['STEAM'],d['ELEC'],d['REF'],d['WWT']])
    C_O=np.array([C_O1,C_O2,C_O3,C_O4,C_O5])
    C_M1=0.035*C_TDC; C_M=np.array([C_M1,0.25*C_M1,1.0*C_M1,0.05*C_M1])
    MOSWB=C_O[0]+C_O[1]+C_M[0]+C_M[1]
    C_OH=np.array([0.071,0.024,0.059,0.074])*MOSWB
    C_prop=0.02*C_TDC
    C_D=np.array([0.08*(C_TDC-1.18*C_alloc),0.06*1.18*C_alloc,0,0])
    COM=C_feed.sum()+C_util.sum()+C_O.sum()+C_M.sum()+C_OH.sum()+C_prop+C_D.sum()
    GE=(0.03+0.01+0.048+0.005+0.02+0.0125)*S
    C_cat_annual = C_cat_inv/cat_year if cat_replace else 0.0
    n=int(round(plife))
    CF_TDC=np.zeros(n+1);CF_WC=np.zeros(n+1);CF_CD=np.zeros(n+1);CF_NE=np.zeros(n+1)
    C_WC=0.0833*COM+0.0192*S+0.0833*S-0.0833*C_feed.sum()
    for i in range(1,n+1):
        oy=i-CONSTR
        C = COM+GE+ (C_cat_annual if oy>=1 else 0.0)
        if i<=CONSTR:
            CF_TDC[i]=-C_TDC/CONSTR
            if i==CONSTR: CF_WC[i]=-C_WC
        if i==CONSTR:
            e=min(i+len(MACRS7),n); CF_CD[i+1:i+1+len(MACRS7)][:e-i]=(C_D.sum()*MACRS7)[:e-i]
        if oy>=1:
            CFC=C-C_D.sum(); CF_NE[i]=(S-CFC-CF_CD[i])*(1-TAX)
        if i==n: CF_WC[i]+=C_WC
    val=0.0
    for i in range(1,n+1):
        dcf=(CF_NE[i]+CF_CD[i])+CF_TDC[i]+CF_WC[i]
        val+=dcf/(1+irate)**i
    return val

def solve_msp(d,**kw):
    f=lambda x: npv(x,d,**kw)
    a,b=0.1,20.0; fa=f(a)
    for _ in range(100):
        m=0.5*(a+b); fm=f(m)
        if fa*fm<=0: b=m
        else: a,fa=m,fm
        if b-a<1e-9: break
    return 0.5*(a+b)

# precompute sum_alloc per case so mult=1 reproduces Excel Allocation
excel_alloc={'j':11843851.359563991,'k':12177185.859252483,'i':12148018.876736287}
for c,d in cases.items():
    C_TBM_base=d['EQ']+20000
    d['sum_alloc']=excel_alloc[c]-0.05*C_TBM_base

if __name__=='__main__':
    print("Baseline check (published model: catalyst replace=False, mult=1):")
    print(f"{'case':4}{'MSP_recon':>12}{'MSP_pub':>12}{'err%':>8}")
    for c,d in cases.items():
        m=solve_msp(d,cat_replace=False)
        print(f"{c:4}{m:12.5f}{d['MSP_pub']:12.5f}{100*(m-d['MSP_pub'])/d['MSP_pub']:8.3f}")
