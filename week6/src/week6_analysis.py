"""Reproducible Week 6 experiments. Run from package root or notebooks folder."""
# %% Setup and continuity
from pathlib import Path
import sys,json,hashlib,platform,joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
BASE=Path.cwd() if (Path.cwd()/'src/healthconnect.py').exists() else Path.cwd().parent
sys.path.insert(0,str(BASE/'src'))
from healthconnect import features,score,BASE_NUM,CAT,REQUIRED
MET=BASE/'outputs/metrics';FIG=BASE/'outputs/figures'
MET.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(BASE/'data/raw/healthconnect_appointment_data.csv',keep_default_na=False)
x=features(raw)
cohort=raw.loc[raw.appointment_outcome.isin(['Attended','No-Show'])].copy()
X=x.loc[cohort.index];y=cohort.appointment_outcome.eq('No-Show').astype(int)
train_mask=X.appointment_date.lt('2026-03-01');test_mask=X.booking_date.ge('2026-03-01')
trainX=X.loc[train_mask];trainy=y.loc[train_mask];testX=X.loc[test_mask];testy=y.loc[test_mask]
old=pd.read_csv(BASE/'reference/week5_test_predictions.csv')
assert old.appointment_id.tolist()==raw.loc[testX.index,'appointment_id'].tolist()
assert len(trainX)==3682 and len(testX)==799
print('Week 5 partitions preserved:',len(trainX),len(testX),'purged',len(X)-len(trainX)-len(testX))
print('Existing March-June comparison set has already been inspected. No fresh holdout claim.')

# %% Predeclared competitors and temporal folds
def make_model(name):
    nums=BASE_NUM.copy();cats=CAT.copy()
    clf=LogisticRegression(C=1,max_iter=2000,random_state=42)
    if name=='No-history logistic':nums=[v for v in nums if v not in ['previous_appointments','prior_no_show_rate','no_prior_history']]
    if name in ['Random forest','Gradient boosting']:
        nums=['age','distance_to_clinic_km','previous_appointments','prior_no_show_rate','no_prior_history','lead_days'];cats=CAT+['weekday']
        if name=='Random forest':clf=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=25,max_features=1.0,random_state=42,n_jobs=1)
        else:clf=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=2,learning_rate=.05,early_stopping=False,random_state=42)
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
       ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    return Pipeline([('prep',prep),('classifier',clf)])
def metrics(truth,prob,t=.5):
    pred=np.asarray(prob)>=t;tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return dict(n=len(truth),roc_auc=roc_auc_score(truth,prob) if len(set(truth))>1 else np.nan,
      average_precision=average_precision_score(truth,prob),brier=brier_score_loss(truth,prob),accuracy=accuracy_score(truth,pred),
      precision=precision_score(truth,pred,zero_division=0),recall=recall_score(truth,pred,zero_division=0),f1=f1_score(truth,pred,zero_division=0),
      tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),flagged=int(pred.sum()),flag_rate=float(pred.mean()))
names=['Week 5 logistic','No-history logistic','Random forest','Gradient boosting']
folds=[('2025-09-01','2025-11-01'),('2025-11-01','2026-01-01'),('2026-01-01','2026-03-01')]
scores=[];out_of_time=[];fold_rows=[]
for k,(start,end) in enumerate(folds,1):
    a=trainX.appointment_date.lt(start);b=trainX.booking_date.ge(start)&trainX.booking_date.lt(end)
    assert trainX.loc[a,'appointment_date'].max()<trainX.loc[b,'booking_date'].min()
    fold_rows.append(dict(fold=k,cutoff=start,booking_end=end,train=int(a.sum()),validation=int(b.sum())))
    for name in names:
        m=make_model(name);m.fit(trainX.loc[a],trainy.loc[a]);prob=m.predict_proba(trainX.loc[b])[:,1]
        scores.append(dict(model=name,fold=k,**metrics(trainy.loc[b],prob)))
        out_of_time.extend(dict(model=name,fold=k,index=int(i),truth=int(trainy.loc[i]),probability=float(p)) for i,p in zip(trainX.loc[b].index,prob))
cv=pd.DataFrame(scores);oof=pd.DataFrame(out_of_time);foldtable=pd.DataFrame(fold_rows)
cv.to_csv(MET/'temporal_validation.csv',index=False);oof.to_csv(MET/'out_of_time_predictions.csv',index=False);foldtable.to_csv(MET/'fold_definitions.csv',index=False)
ranking=cv.groupby('model').agg(mean_auc=('roc_auc','mean'),auc_sd=('roc_auc','std'),mean_brier=('brier','mean')).sort_values(['mean_auc','mean_brier'],ascending=[False,True])
candidate=ranking.index[0]
ranking.to_csv(MET/'validation_ranking.csv');print(foldtable.to_string(index=False));print(ranking.round(4));print('Candidate selected using internal validation:',candidate)

# %% Operating threshold and uncertainty sensitivity
selected=oof.loc[oof.model.eq(candidate)].copy()
# Illustrative staffing scenario: maximum 30 percent of validation records flagged.
# This is a modelling scenario, not a clinic-approved capacity requirement.
thresholds=[]
for threshold in np.linspace(0,1,201):thresholds.append(dict(threshold=float(threshold),**metrics(selected.truth,selected.probability,threshold)))
threshold_table=pd.DataFrame(thresholds);eligible=threshold_table.loc[threshold_table.flag_rate.le(.30)]
threshold=float(eligible.sort_values(['recall','precision','threshold'],ascending=[False,False,True]).iloc[0].threshold)
threshold_table.to_csv(MET/'threshold_scenarios.csv',index=False)
sensitivity=[]
for flag in [False,True]:
    for k,(start,end) in enumerate(folds,1):
        a=trainX.appointment_date.lt(start);b=trainX.booking_date.ge(start)&trainX.booking_date.lt(end)
        # Assess the same non-Sunday validation population under two training policies.
        b &= trainX.appointment_date.dt.dayofweek.ne(6)
        if flag:a &= trainX.appointment_date.dt.dayofweek.ne(6)
        m=make_model(candidate);m.fit(trainX.loc[a],trainy.loc[a]);pr=m.predict_proba(trainX.loc[b])[:,1]
        sensitivity.append(dict(exclude_sunday_training=flag,fold=k,**metrics(trainy.loc[b],pr)))
pd.DataFrame(sensitivity).to_csv(MET/'sunday_sensitivity.csv',index=False)
print('Illustrative 30% validation-workload threshold:',threshold)
print(threshold_table.loc[threshold_table.threshold.eq(threshold)].round(4).to_string(index=False))

# %% Reused comparison set and preserved baseline
fitted={};comparisons=[];predframe=cohort.loc[testX.index,['appointment_id','patient_id','gender','age_group','appointment_type']].copy();predframe['truth']=testy
for name in names:
    m=make_model(name);m.fit(trainX,trainy);fitted[name]=m;pr=m.predict_proba(testX)[:,1]
    comparisons.append(dict(model=name,threshold=.5,**metrics(testy,pr)))
    predframe[name]=pr
np.testing.assert_allclose(predframe['Week 5 logistic'],old['Logistic regression_probability'],atol=1e-12)
prob=predframe[candidate].to_numpy();comparisons.append(dict(model=candidate+' capacity scenario',threshold=threshold,**metrics(testy,prob,threshold)))
comparison=pd.DataFrame(comparisons);comparison.to_csv(MET/'reused_test_comparison.csv',index=False)
predframe['candidate_prediction']=(prob>=.5).astype(int)
predframe['error_type']=np.select([(testy==1)&(prob<.5),(testy==0)&(prob>=.5)],['False negative','False positive'],default='Correct')
predframe.to_csv(MET/'reused_test_predictions.csv',index=False)
print(comparison.round(4).to_string(index=False))

# %% Error analysis using observed segments, not causal claims
errors=[]
for name in ['Week 5 logistic',candidate]:
    for field in ['gender','age_group','appointment_type']:
        for group,part in predframe.groupby(field):
            met=metrics(part.truth,part[name]);met['false_positive_rate']=met['fp']/(met['fp']+met['tn']) if met['fp']+met['tn'] else np.nan
            errors.append(dict(model=name,field=field,group=group,**met))
error_table=pd.DataFrame(errors).drop_duplicates();error_table.to_csv(MET/'segment_error_analysis.csv',index=False)
detail=predframe[['appointment_id','error_type','truth']].join(testX[['lead_days','prior_no_show_rate','age','distance_to_clinic_km']])
detail.groupby('error_type').agg(n=('truth','size'),lead_median=('lead_days','median'),prior_rate_median=('prior_no_show_rate','median'),age_median=('age','median')).to_csv(MET/'error_profiles.csv')
print(pd.read_csv(MET/'error_profiles.csv').to_string(index=False))
print('Segment comparisons are descriptive; no causal explanation or proven fairness.')

# %% Serialize the selected preprocessing and model together
obj=dict(pipeline=fitted[candidate],threshold=threshold,version='healthconnect-week6-v1',candidate=candidate,
         requires_history=candidate!='No-history logistic',known_categories={c:sorted(trainX[c].unique().tolist()) for c in CAT})
(BASE/'models').mkdir(exist_ok=True);joblib.dump(obj,BASE/'models/candidate_pipeline.joblib')
input_columns=REQUIRED if obj['requires_history'] else [c for c in REQUIRED if c not in ['previous_appointments','previous_no_shows']]
sample=raw.loc[testX.index[:5],input_columns];sample.to_csv(BASE/'integration/sample_input.csv',index=False)
scored=score(sample,BASE/'models/candidate_pipeline.joblib');scored.to_csv(BASE/'integration/sample_output.csv',index=False)
np.testing.assert_allclose(scored.no_show_probability,fitted[candidate].predict_proba(features(sample,require_history=obj['requires_history']))[:,1],atol=1e-12)
print(scored.to_string(index=False))

# %% Evidence figures and run metadata
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,4));ranking.mean_auc.plot.barh(ax=ax,color='#C69A4B',xerr=ranking.auc_sd)
ax.set(xlabel='Mean ROC-AUC across three temporal folds',title='Candidate selection on Week 5 training records',xlim=(.45,.85));fig.tight_layout();fig.savefig(FIG/'01_validation_comparison.png',dpi=170);plt.show()
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
for i,name in enumerate(['Week 5 logistic',candidate]):
    cm=confusion_matrix(testy,predframe[name]>=.5,labels=[0,1]);axes[i].imshow(cm,cmap='Greys')
    for a in range(2):
        for b in range(2):axes[i].text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>(cm.min()+cm.max())/2 else 'black')
    axes[i].set(xticks=[0,1],yticks=[0,1],xticklabels=['Attend','No-show'],yticklabels=['Attend','No-show'],xlabel='Predicted',ylabel='Actual',title=name)
fig.suptitle('Reused Week 5 comparison set at threshold 0.5');fig.tight_layout();fig.savefig(FIG/'02_error_comparison.png',dpi=170);plt.show()
fig,ax=plt.subplots(figsize=(8,4));ax.plot(threshold_table.threshold,threshold_table.recall,label='Recall');ax.plot(threshold_table.threshold,threshold_table.flag_rate,label='Flagged share',color='#C69A4B');ax.axvline(threshold,ls='--',color='gray',label=f'Scenario threshold {threshold:.3f}');ax.set(xlabel='Threshold',ylabel='Proportion',title='Internal validation workload and recall');ax.legend();fig.tight_layout();fig.savefig(FIG/'03_threshold_tradeoff.png',dpi=170);plt.show()
summary=dict(candidate=candidate,threshold=threshold,raw_sha256=hashlib.sha256((BASE/'data/raw/healthconnect_appointment_data.csv').read_bytes()).hexdigest(),
    train=len(trainX),reused_test=len(testX),folds=fold_rows,validation=ranking.reset_index().to_dict('records'),comparison=comparison.to_dict('records'),
    python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,
    integration='AI-simulated ML Engineering counterpart, requested by user; no human intern exchange claimed',
    evaluation_limit='Old test reused descriptively; candidate and threshold selected internally. Selection validation is not an unbiased final estimate.')
(MET/'run_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
