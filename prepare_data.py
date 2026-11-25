from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nilearn.datasets import fetch_abide_pcp

def main():
    out=Path('data/processed'); out.mkdir(parents=True,exist_ok=True)
    ds=fetch_abide_pcp(data_dir='./data',pipeline='cpac',band_pass_filtering=True,global_signal_regression=False,derivatives=['rois_cc200'],quality_checked=True,n_subjects=None)
    meta=pd.DataFrame(ds.phenotypic).reset_index(drop=True)
    assert len(meta)==len(ds.rois_cc200)
    matrices=[]; rows=[]; rejected=[]; seen=set()
    for i,(ts,row) in enumerate(zip(ds.rois_cc200,meta.to_dict('records'))):
        x=np.asarray(ts,dtype=float)
        reason=None
        if row['DX_GROUP'] not in (1,2): reason='invalid label'
        elif pd.isna(row['SUB_ID']) or row['SUB_ID'] in seen: reason='invalid ID'
        elif x.ndim!=2 or x.shape[1]!=200 or x.shape[0]<30: reason='invalid shape'
        elif not np.isfinite(x).all() or np.any(x.std(axis=0)<1e-10): reason='invalid signal'
        if reason: rejected.append({'index':i,'reason':reason}); continue
        c=np.corrcoef(x,rowvar=False)
        if not np.isfinite(c).all(): rejected.append({'index':i,'reason':'invalid correlation'}); continue
        z=np.arctanh(np.clip(c,-.999999,.999999)); np.fill_diagonal(z,0)
        matrices.append(z.astype('float32')); rows.append(row); seen.add(row['SUB_ID'])
    if not rows: raise ValueError('No valid participants')
    m=pd.DataFrame(rows); X=np.stack(matrices); y=(m.DX_GROUP.to_numpy()==1).astype(int)
    m.to_csv(out/'participants.csv',index=False)
    pd.DataFrame(rejected,columns=['index','reason']).to_csv(out/'rejected.csv',index=False)
    counts=pd.crosstab(m.SITE_ID,m.DX_GROUP); counts.to_csv(out/'site_counts.csv')
    print('Downloaded:',len(meta),'Accepted:',len(m),'Rejected:',len(rejected)); print(counts)
    if len(m)<40 or min(np.bincount(y,minlength=2))<10: raise ValueError('Too few participants')
    tr,te=train_test_split(np.arange(len(y)),test_size=.15,stratify=y,random_state=42)
    tr,va=train_test_split(tr,test_size=.15/.85,stratify=y[tr],random_state=42)
    np.savez_compressed(out/'dataset.npz',X=X,y=y,train=tr,val=va,test=te)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,a,title in zip(axes,[np.tanh(X[tr[0]]),np.tanh(X[tr][y[tr]==1]).mean(0),np.tanh(X[tr][y[tr]==0]).mean(0)],['Example','ASD train mean','Control train mean']):
        np.fill_diagonal(a,1); im=ax.imshow(a,cmap='coolwarm',vmin=-1,vmax=1); ax.set_title(title)
    fig.colorbar(im,ax=axes.tolist()); fig.savefig(out/'connectivity.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    print('Preparation complete:',X.shape)
if __name__=='__main__': main()
