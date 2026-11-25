from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

def main():
    d=np.load('data/processed/dataset.npz'); X=d['X']; y=d['y']; meta=pd.read_csv('data/processed/participants.csv'); assert len(meta)==len(y); assert np.array_equal(y,(meta.DX_GROUP.to_numpy()==1).astype(int))
    a,b=np.triu_indices(X.shape[1],1); F=X[:,a,b]; sites=meta.SITE_ID.to_numpy(); rows=[]
    # All eligible sites, no cherry picking. Compare SVM with original random-split SVM.
    for site in sorted(set(sites)):
        te=sites==site; tr=~te
        if min(np.bincount(y[te],minlength=2))<5: continue
        model=make_pipeline(StandardScaler(),SVC()); model.fit(F[tr],y[tr]); score=model.decision_function(F[te]); rows.append({'site':site,'n':int(te.sum()),'auc':roc_auc_score(y[te],score),'accuracy':accuracy_score(y[te],model.predict(F[te]))}); print(rows[-1],flush=True)
    Path('analysis').mkdir(exist_ok=True); pd.DataFrame(rows).to_csv('analysis/site_holdout_svm.csv',index=False)
if __name__=='__main__': main()
