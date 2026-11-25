from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,confusion_matrix
import joblib

def main():
    tf.keras.utils.set_random_seed(42)
    d=np.load('data/processed/dataset.npz'); X,y=d['X'],d['y']; tr,va,te=[d[k] for k in ('train','val','test')]
    out=Path('data/results'); out.mkdir(parents=True,exist_ok=True)
    models=Path('saved_models'); models.mkdir(exist_ok=True)
    a,b=np.triu_indices(X.shape[1],1); raw=X[:,a,b]
    scaler=StandardScaler().fit(raw[tr]); F=scaler.transform(raw).astype('float32')
    pca=PCA(n_components=min(128,len(tr)-1,F.shape[1]),random_state=42).fit(F[tr]); P=pca.transform(F).astype('float32')
    joblib.dump(scaler,models/'scaler.joblib'); joblib.dump(pca,models/'pca.joblib')
    metrics=[]; predictions={'index':te,'label':y[te]}
    def record(name,score):
        score=np.asarray(score).ravel(); pred=(score>=.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(y[te],pred,labels=[0,1]).ravel()
        metrics.append(dict(model=name,auc=roc_auc_score(y[te],score),macro_f1=f1_score(y[te],pred,average='macro'),accuracy=accuracy_score(y[te],pred),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp)))
        predictions[name]=score
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False); pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    for name,model in [('Dummy',DummyClassifier(strategy='prior')),('SVM',SVC(probability=True,random_state=42))]:
        model.fit(F[tr],y[tr]); record(name,model.predict_proba(F[te])[:,1]); joblib.dump(model,models/(name+'.joblib'))
    L=tf.keras.layers
    def mlp(n,drop): return tf.keras.Sequential([L.Input((n,)),L.Dense(128,activation='relu'),L.Dropout(drop),L.Dense(64,activation='relu'),L.Dropout(drop),L.Dense(1,activation='sigmoid')])
    cnn=tf.keras.Sequential([L.Input((200,200,1)),L.Conv2D(16,3,activation='relu',padding='same'),L.MaxPooling2D(),L.Conv2D(32,3,activation='relu'),L.MaxPooling2D(),L.GlobalAveragePooling2D(),L.Dense(32,activation='relu'),L.Dropout(.5),L.Dense(1,activation='sigmoid')])
    for name,model,features in [('MLP',mlp(F.shape[1],.5),F),('MLP_no_dropout',mlp(F.shape[1],0),F),('PCA_MLP',mlp(P.shape[1],.5),P),('CNN',cnn,X[...,None])]:
        model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='auc')])
        h=model.fit(features[tr],y[tr],validation_data=(features[va],y[va]),epochs=100,batch_size=32,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=2)
        pd.DataFrame(h.history).to_csv(out/(name+'_history.csv'),index=False); model.save(models/(name+'.keras')); record(name,model.predict(features[te],verbose=0))
    print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__': main()
