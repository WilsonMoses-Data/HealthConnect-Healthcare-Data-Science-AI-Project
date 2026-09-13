"""Executable cross-track validation evidence. Run from package root."""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from healthconnect import score
artifact=ROOT/'models/candidate_pipeline.joblib'
sample=pd.read_csv(ROOT/'integration/sample_input.csv',keep_default_na=False)
results=[]
def case(name,fn):
    try:fn();results.append(dict(test=name,status='PASS'))
    except Exception as e:results.append(dict(test=name,status='FAIL',detail=str(e)))
def valid():
    a=score(sample,artifact);assert len(a)==len(sample);assert a.appointment_id.tolist()==sample.appointment_id.tolist();assert a.no_show_probability.between(0,1).all();assert set(a.support_flag)<={0,1}
def rejects(frame):
    try:score(frame,artifact)
    except ValueError:return
    raise AssertionError('Expected input rejection')
def changed(col,value):
    d=sample.copy();d[col]=d[col].astype(object);d.loc[d.index[0],col]=value;return d
case('Valid batch preserves IDs and bounds',valid)
case('Missing required field rejected',lambda:rejects(sample.drop(columns='age')))
case('Duplicate IDs rejected',lambda:rejects(pd.concat([sample,sample.iloc[:1]],ignore_index=True)))
case('Invalid date rejected',lambda:rejects(changed('booking_date','not-a-date')))
case('Appointment before booking rejected',lambda:rejects(changed('appointment_date','01/01/2020')))
case('Negative distance rejected',lambda:rejects(changed('distance_to_clinic_km',-1)))
case('Missing age rejected',lambda:rejects(changed('age',np.nan)))
case('Out-of-scope age rejected',lambda:rejects(changed('age',17)))
case('Infinite distance rejected',lambda:rejects(changed('distance_to_clinic_km',np.inf)))
case('Empty batch rejected',lambda:rejects(sample.iloc[:0]))
case('Blank appointment type rejected',lambda:rejects(changed('appointment_type','')))
def missing_distance():
    r=score(changed('distance_to_clinic_km',''),artifact);assert 'distance_imputed' in r.iloc[0].warnings
case('Missing distance imputed and flagged',missing_distance)
def unknown():
    r=score(changed('appointment_type','New category'),artifact);assert 'unknown_category' in r.iloc[0].warnings
case('Unknown category supported and flagged',unknown)
def no_history():
    r=score(sample.drop(columns=['previous_appointments','previous_no_shows'],errors='ignore'),artifact);assert len(r)==len(sample)
case('Selected candidate needs no history columns',no_history)
def invariant():
    a=score(sample,artifact);d=sample.copy();d['appointment_outcome']='No-Show';d['patient_id']='ignored';d['previous_no_shows']='unused'
    np.testing.assert_array_equal(a.no_show_probability,score(d,artifact).no_show_probability)
case('Outcome IDs and unused history cannot alter scores',invariant)
def reorder():
    a=score(sample,artifact).set_index('appointment_id').sort_index();b=score(sample.iloc[::-1],artifact).set_index('appointment_id').sort_index();np.testing.assert_array_equal(a.no_show_probability,b.no_show_probability)
case('Row reordering preserves individual scores',reorder)
def batches():
    a=score(sample,artifact);b=pd.concat([score(sample.iloc[[i]],artifact) for i in range(len(sample))]);np.testing.assert_allclose(a.no_show_probability,b.no_show_probability,atol=1e-12)
case('Single and batch scoring agree',batches)
pd.DataFrame(results).to_csv(ROOT/'integration/test_results.csv',index=False)
print(json.dumps(results,indent=2));assert all(r['status']=='PASS' for r in results)
