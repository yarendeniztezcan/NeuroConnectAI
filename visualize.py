from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def main():
    import tensorflow as tf
    d=np.load('data/processed/dataset.npz'); X=d['X']; y=d['y']; tr=d['train']; va=d['val']
    out=Path('visualizations'); out.mkdir(exist_ok=True)
    a,b=np.triu_indices(X.shape[1],1)
    # Group summaries use training participants only; Fisher-z mean difference.
    diff=X[tr[y[tr]==1]].mean(0)-X[tr[y[tr]==0]].mean(0)
    lim=max(float(np.abs(diff).max()),1e-8)
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(diff,cmap='coolwarm',vmin=-lim,vmax=lim); fig.colorbar(im,ax=ax,label='Mean Fisher-z difference'); ax.set(title='ASD minus control (training set)',xlabel='ROI index',ylabel='ROI index'); fig.savefig(out/'group_difference.png',dpi=160); plt.close(fig)
    scaler=joblib.load('saved_models/scaler.joblib'); model=tf.keras.models.load_model('saved_models/MLP.keras',compile=False)
    # Gradient times input, relative to zero standardized input (training mean).
    f=scaler.transform(X[va][:,a,b]).astype('float32'); t=tf.convert_to_tensor(f)
    with tf.GradientTape() as tape:
        tape.watch(t); scores=model(t,training=False)
    grad=tape.gradient(scores,t).numpy(); importance=np.mean(np.abs(grad*f),axis=0)
    idx=np.argsort(importance)[-20:][::-1]
    edges=pd.DataFrame({'roi_a':a[idx]+1,'roi_b':b[idx]+1,'importance':importance[idx]}); edges.to_csv(out/'important_edges.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,7)); ax.barh([f'{i}–{j}' for i,j in zip(edges.roi_a,edges.roi_b)][::-1],edges.importance[::-1]); ax.set(xlabel='Mean absolute gradient × input',title='MLP attribution — validation set'); fig.tight_layout(); fig.savefig(out/'importance.png',dpi=160); plt.close(fig)
    n=X.shape[1]; theta=np.linspace(0,2*np.pi,n,endpoint=False); xy=np.column_stack([np.cos(theta),np.sin(theta)])
    fig,ax=plt.subplots(figsize=(8,8)); ax.scatter(*xy.T,s=8,color='gray');
    for i,j in zip(a[idx],b[idx]): ax.plot(*xy[[i,j]].T,alpha=.65)
    for i in np.unique(np.r_[a[idx],b[idx]]): ax.text(*(xy[i]*1.08),str(i+1),fontsize=7,ha='center',va='center')
    ax.set(aspect='equal',title='Top 20 attributed connections (ROI order)'); ax.axis('off'); fig.savefig(out/'connectogram.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    # Anatomical coordinates must be independently verified against time-series column order.
    coords=Path('data/cc200_coordinates.csv')
    if coords.exists():
        from nilearn import plotting
        c=pd.read_csv(coords); assert list(c['roi'])==list(range(1,n+1)), 'ROI order mismatch'
        xyz=c[['x','y','z']].to_numpy(); assert np.isfinite(xyz).all()
        adj=np.zeros((n,n)); adj[a[idx],b[idx]]=importance[idx]; adj+=adj.T
        plotting.view_connectome(adj,xyz,edge_threshold=None,symmetric_cmap=False).save_as_html(str(out/'brain.html'))
    else: print('3D skipped: verified data/cc200_coordinates.csv required')
    print('Visualization complete')
if __name__=='__main__': main()
