import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
OUT='output'
pw = pd.read_csv(f'{OUT}/playerweek.csv'); POS=['QB','RB','WR','TE']
pw['resid']=pw.ppr-pw.xfp
def rel(d, col, lam):
    tgm = d.groupby(['week','team','opp','pos'])[col].sum().reset_index(); out={}
    for p in POS:
        t=tgm[tgm.pos==p]
        X=pd.get_dummies(t[['team','opp']],prefix=['o','d']).astype(float)
        m=Ridge(alpha=lam).fit(X,t[col].values); co=pd.Series(m.coef_,index=X.columns)
        out[p]={c[2:]:v/t[col].mean() for c,v in co.items() if c.startswith('d_')}
    return out
print('Persistence: corr(defense softness weeks<=c, softness weeks>c), per pos; mean over seasons/cutoffs')
for col in ['ppr','xfp','resid']:
    for lam in [1,5,20]:
        rs={p:[] for p in POS}
        for s in [2024,2025]:
            for c in [4,6,8]:
                d=pw[pw.season==s]; a=rel(d[d.week<=c],col,lam) if col!='resid' else None
                if col=='resid':
                    # residual allowed per team-game (efficiency luck allowed)
                    a=rel(d[d.week<=c],col,lam); b=rel(d[(d.week>c)&(d.week<=17)],col,lam)
                else:
                    b=rel(d[(d.week>c)&(d.week<=17)],col,lam)
                for p in POS:
                    k=sorted(set(a[p])&set(b[p])); rs[p].append(np.corrcoef([a[p][x] for x in k],[b[p][x] for x in k])[0,1])
        print(col,'lam',lam,{p:round(float(np.mean(v)),2) for p,v in rs.items()})
