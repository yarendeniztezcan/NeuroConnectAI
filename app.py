from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title='NeuroConnectAI',layout='wide')
st.title('NeuroConnectAI | Research dashboard')
st.caption('ABIDE I — Research use only. Not a clinical diagnostic tool.')
tabs=st.tabs(['Models','Brain connectivity','Explainability','Research prediction'])
with tabs[0]:
    p=Path('data/results/metrics.csv')
    if not p.exists(): p=Path('analysis/metrics.csv')
    if p.exists(): st.dataframe(pd.read_csv(p)); st.bar_chart(pd.read_csv(p).set_index('model')['auc'])
    for p in [Path('analysis/roc.png'),*sorted(Path('analysis').glob('confusion_*.png'))]:
        if p.exists(): st.image(str(p),caption=p.stem)
    p=Path('analysis/site_holdout_svm.csv')
    if p.exists(): st.dataframe(pd.read_csv(p))
with tabs[1]:
    for name in ['group_difference.png','connectogram.png']:
        p=Path('visualizations')/name
        if p.exists(): st.image(str(p))
    p=Path('visualizations/brain.html')
    if p.exists(): st.components.v1.html(p.read_text(encoding='utf-8'),height=650,scrolling=True)
    else: st.info('Anatomical view awaits verified CC200 coordinates.')
with tabs[2]:
    p=Path('visualizations/importance.png')
    if p.exists(): st.image(str(p))
    p=Path('visualizations/important_edges.csv')
    if p.exists(): st.dataframe(pd.read_csv(p))
    st.caption('Gradient × input attribution; not causal or diagnostic evidence.')
with tabs[3]:
    st.warning('Only C-PAC, band-pass filtered, no-GSR CC200 time series in original ROI order. Do not upload identifiable or clinical data.')
    f=st.file_uploader('ROI time series (.1D)',type=['1D','txt'])
    if f and st.button('Compute research score'):
        try:
            import tensorflow as tf
            import joblib
            x=np.loadtxt(f); assert x.ndim==2 and x.shape[1]==200 and x.shape[0]>=30 and np.isfinite(x).all() and (x.std(0)>1e-10).all(), 'Invalid ROI time series'
            c=np.corrcoef(x,rowvar=False); z=np.arctanh(np.clip(c,-.999999,.999999)); a,b=np.triu_indices(200,1)
            scaler=joblib.load('saved_models/scaler.joblib'); model=tf.keras.models.load_model('saved_models/MLP.keras',compile=False); score=float(model.predict(scaler.transform(z[a,b][None,:]),verbose=0).ravel()[0]); st.metric('MLP ASD-class score (uncalibrated)',f'{score:.3f}')
        except Exception as e: st.error(str(e))
