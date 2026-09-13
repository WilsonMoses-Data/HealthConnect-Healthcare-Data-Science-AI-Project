"""Week 6 raw-input contract, feature engineering and batch scoring."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

REQUIRED = ['appointment_id','booking_date','appointment_date','age','distance_to_clinic_km',
            'previous_appointments','previous_no_shows','appointment_type','appointment_time']
BASE_NUM = ['age','distance_to_clinic_km','previous_appointments','prior_no_show_rate',
            'no_prior_history','is_weekend','log_lead_days']
CAT = ['appointment_type','appointment_time']

def features(frame, require_history=True):
    """Validate raw columns; derive booking-known features without using outcomes."""
    required=REQUIRED if require_history else [c for c in REQUIRED if c not in ['previous_appointments','previous_no_shows']]
    missing = sorted(set(required)-set(frame.columns))
    if missing: raise ValueError(f'Missing required columns: {missing}')
    if frame.empty: raise ValueError('Empty batch')
    x = frame[required].copy()
    if not require_history:
        x['previous_appointments']=0
        x['previous_no_shows']=0
    if x.appointment_id.isna().any() or x.appointment_id.astype(str).str.strip().eq('').any():
        raise ValueError('Appointment IDs must be nonempty')
    if x.appointment_id.duplicated().any(): raise ValueError('Duplicate appointment IDs')
    for col in ['booking_date','appointment_date']:
        x[col] = pd.to_datetime(x[col],format='%m/%d/%Y',errors='coerce')
        if x[col].isna().any(): raise ValueError(f'Invalid {col}; use month/day/year')
    if x.appointment_date.lt(x.booking_date).any(): raise ValueError('Appointment precedes booking')
    for col in ['age','distance_to_clinic_km','previous_appointments','previous_no_shows']:
        values = x[col].where(x[col].ne(''),np.nan)
        parsed = pd.to_numeric(values,errors='coerce')
        if (values.notna() & parsed.isna()).any() or np.isinf(parsed).any(): raise ValueError(f'Invalid numeric {col}')
        if parsed.lt(0).any(): raise ValueError(f'Negative {col}')
        if col!='distance_to_clinic_km' and parsed.isna().any(): raise ValueError(f'Missing {col}')
        x[col]=parsed
    if not x.age.between(18,80).all(): raise ValueError('Age outside validated prototype range 18-80')
    for col in ['previous_appointments','previous_no_shows']:
        if x[col].mod(1).ne(0).any(): raise ValueError('History counts must be whole numbers')
    if x.previous_no_shows.gt(x.previous_appointments).any(): raise ValueError('No-shows exceed history')
    for col in CAT:
        if x[col].isna().any() or x[col].astype(str).str.strip().eq('').any(): raise ValueError(f'Missing {col}')
        x[col]=x[col].astype(str).str.strip()
    x['prior_no_show_rate']=np.divide(x.previous_no_shows,x.previous_appointments,
        out=np.zeros(len(x)),where=x.previous_appointments.to_numpy()!=0)
    x['no_prior_history']=x.previous_appointments.eq(0).astype(int)
    x['is_weekend']=x.appointment_date.dt.dayofweek.ge(5).astype(int)
    x['weekday']=x.appointment_date.dt.day_name()
    x['lead_days']=(x.appointment_date-x.booking_date).dt.days
    x['log_lead_days']=np.log1p(x.lead_days)
    return x

def score(raw,artifact):
    """Score with a trusted local artifact; no fitting or remote actions."""
    obj=joblib.load(artifact)
    x=features(raw, require_history=obj['requires_history'])
    probability=obj['pipeline'].predict_proba(x)[:,1]
    if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
        raise RuntimeError('Invalid model probabilities')
    warning=[]
    for _,row in x.iterrows():
        flags=['experimental_model']
        if obj['requires_history']:flags.append('history_timing_unverified')
        if row.appointment_date.dayofweek==6:flags.append('sunday_schedule_conflict')
        if pd.isna(row.distance_to_clinic_km):flags.append('distance_imputed')
        if any(row[col] not in obj['known_categories'][col] for col in CAT):flags.append('unknown_category')
        warning.append(';'.join(flags))
    return pd.DataFrame({'appointment_id':x.appointment_id.to_numpy(),'no_show_probability':probability,
        'support_flag':(probability>=obj['threshold']).astype(int),'threshold':obj['threshold'],
        'model_version':obj['version'],'warnings':warning})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=score(pd.read_csv(a.input,keep_default_na=False),a.model)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    print(f'Scored {len(result)} records. Output: {a.output}')
