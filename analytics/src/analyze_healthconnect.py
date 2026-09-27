from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/raw/HealthConnect_Appointment_Data.csv'
OUT=ROOT/'analytics/outputs'
FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
FIG.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(SOURCE,keep_default_na=False)
d=df.copy()
for col in ['distance_to_clinic_km','waiting_time_minutes']:
    d[col]=pd.to_numeric(d[col].replace('',np.nan),errors='coerce')
for col in ['booking_date','appointment_date']:
    d[col]=pd.to_datetime(d[col],errors='coerce')
for col in ['booking_lead_days','previous_appointments','previous_no_shows']:
    d[col]=pd.to_numeric(d[col],errors='coerce')
d['eligible']=d.appointment_outcome.isin(['Attended','No-Show'])
d['no_show']=d.appointment_outcome.eq('No-Show')
d['lead_band']=pd.cut(d.booking_lead_days,[-1,3,7,14,30,np.inf],
    labels=['0-3 days','4-7 days','8-14 days','15-30 days','31+ days'])
completed=d[d.eligible].copy()
n=len(d); patients=d.patient_id.nunique()
att=int(d.appointment_outcome.eq('Attended').sum())
ns=int(d.appointment_outcome.eq('No-Show').sum())
cancelled=int(d.appointment_outcome.eq('Cancelled').sum())
done=att+ns
lead=completed.groupby('lead_band',observed=False).no_show.agg(['sum','count','mean']).reset_index()
lead.columns=['lead_band','no_show_count','completed_appointments','no_show_rate']
rem=completed.groupby('reminder_sent').no_show.agg(['sum','count','mean']).reset_index()
rem.columns=['reminder_sent','no_show_count','completed_appointments','no_show_rate']
appt_type=completed.groupby('appointment_type').no_show.agg(['sum','count','mean']).reset_index()
appt_type.columns=['appointment_type','no_show_count','completed_appointments','no_show_rate']

kpis=[
 ('Appointment records',n,n,'All source rows','records'),
 ('Distinct patients',patients,n,'Distinct anonymized patient identifiers','patients'),
 ('Completed appointments',done,n,'Attended + No-Show','records'),
 ('Attendance rate',att,done,'Attended / (Attended + No-Show)','rate'),
 ('No-show rate',ns,done,'No-Show / (Attended + No-Show)','rate'),
 ('Cancellation rate',cancelled,n,'Cancelled / all appointment records','rate'),
 ('Reminder coverage',int(d.reminder_sent.eq('Yes').sum()),n,'Reminder Sent = Yes / all appointment records','rate')]
pd.DataFrame([{'metric':name,'definition':definition,'numerator':num,'denominator':den,
    'value':num/den if unit=='rate' else num,'unit':unit}
    for name,num,den,definition,unit in kpis]).to_csv(OUT/'kpi_summary.csv',index=False)
lead.to_csv(OUT/'lead_time_findings.csv',index=False)
rem.to_csv(OUT/'reminder_findings.csv',index=False)
appt_type.to_csv(OUT/'appointment_type_findings.csv',index=False)
segments=[]
for column,label in [('reminder_sent','Reminder status'),('appointment_type','Appointment type'),
                     ('appointment_day','Appointment day'),('reminder_channel','Reminder channel')]:
    t=completed.groupby(column).no_show.agg(['sum','count','mean']).reset_index()
    for _,r in t.iterrows():
        segments.append({'segment_type':label,'segment':str(r[column]),
            'no_show_count':int(r['sum']),'completed_appointments':int(r['count']),
            'no_show_rate':float(r['mean'])})
pd.DataFrame(segments).to_csv(OUT/'segment_findings.csv',index=False)

lead_days=(d.appointment_date-d.booking_date).dt.days
weekday=d.appointment_date.dt.day_name()
quality=[
 ('Rows',n,'5,000 rows in source','PASS' if n==5000 else 'REVIEW'),
 ('Unique appointment IDs',int(d.appointment_id.nunique()),'Must equal row count','PASS' if d.appointment_id.is_unique else 'FAIL'),
 ('Exact duplicate rows',int(d.duplicated().sum()),'0 expected','PASS' if not d.duplicated().any() else 'REVIEW'),
 ('Invalid booking or appointment dates',int(d.booking_date.isna().sum()+d.appointment_date.isna().sum()),'0 expected','PASS' if d.booking_date.notna().all() and d.appointment_date.notna().all() else 'FAIL'),
 ('Lead-day/date mismatches',int((lead_days!=d.booking_lead_days).sum()),'0 expected','PASS' if (lead_days==d.booking_lead_days).all() else 'REVIEW'),
 ('Weekday/date mismatches',int((weekday!=d.appointment_day).sum()),'0 expected','PASS' if (weekday==d.appointment_day).all() else 'REVIEW'),
 ('Previous no-shows above prior appointments',int((d.previous_no_shows>d.previous_appointments).sum()),'0 expected','PASS' if (d.previous_no_shows<=d.previous_appointments).all() else 'FAIL'),
 ('Missing distance values',int(d.distance_to_clinic_km.isna().sum()),'Limited missingness noted in dictionary','DOCUMENTED'),
 ('Missing waiting time values',int(d.waiting_time_minutes.isna().sum()),'Timing is unclear','DOCUMENTED'),
 ('Repeated patient IDs',int((d.patient_id.value_counts()>1).sum()),f'{patients:,} distinct patients across {n:,} appointments','DOCUMENTED')]
pd.DataFrame(quality,columns=['check','observed','expected_or_note','status']).to_csv(OUT/'data_quality_checks.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#D9D9D9','text.color':'#20242A',
    'axes.labelcolor':'#555B63','xtick.color':'#555B63','ytick.color':'#555B63'})
gold='#B18A45'; dark='#17191D'
def outcome_chart():
    values=d.appointment_outcome.value_counts().reindex(['Attended','No-Show','Cancelled'])
    fig,ax=plt.subplots(figsize=(8,3.7))
    bars=ax.barh(['Attended','No-show','Cancelled'],values.values,color=[gold,dark,'#AEB3BA'])
    ax.invert_yaxis(); ax.set_xlabel('Appointment records'); ax.set_title('Appointment outcomes',loc='left',pad=13)
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True)
    for b,v in zip(bars,values.values):
        ax.text(v+30,b.get_y()+b.get_height()/2,f'{v:,}  ({v/n:.1%})',va='center',fontsize=9)
    ax.set_xlim(0,max(values.values)*1.34); fig.tight_layout()
    fig.savefig(FIG/'outcome_mix.png',dpi=180,bbox_inches='tight'); plt.close(fig)
def lead_chart():
    fig,ax=plt.subplots(figsize=(8,3.8))
    ax.bar(lead.lead_band.astype(str),lead.no_show_rate*100,color=[gold,gold,gold,gold,dark])
    ax.set_ylim(0,76); ax.set_ylabel('No-show rate among completed appointments (%)')
    ax.set_title('No-show rate by booking lead time',loc='left',pad=13)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    for i,r in lead.iterrows():
        ax.text(i,r.no_show_rate*100+1.3,f'{r.no_show_rate:.1%}\nn={int(r.completed_appointments):,}',ha='center',fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/'lead_time_no_show.png',dpi=180,bbox_inches='tight'); plt.close(fig)
def reminder_chart():
    x=rem.set_index('reminder_sent').reindex(['Yes','No'])
    fig,ax=plt.subplots(figsize=(8,3.5))
    bars=ax.bar(['Reminder recorded','No reminder recorded'],x.no_show_rate*100,color=[gold,dark],width=.55)
    ax.set_ylim(0,70); ax.set_ylabel('No-show rate among completed appointments (%)')
    ax.set_title('Observed rates by reminder status',loc='left',pad=13)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    for b,(_,r) in zip(bars,x.iterrows()):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.2,
            f'{r.no_show_rate:.1%}  |  n={int(r.completed_appointments):,}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'reminder_association.png',dpi=180,bbox_inches='tight'); plt.close(fig)
def type_chart():
    x=appt_type.sort_values('no_show_rate')
    fig,ax=plt.subplots(figsize=(8,3.7))
    bars=ax.barh(x.appointment_type,x.no_show_rate*100,color=[gold,gold,gold,dark])
    ax.set_xlim(0,70); ax.set_xlabel('No-show rate among completed appointments (%)')
    ax.set_title('Observed rate by appointment type',loc='left',pad=13)
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True)
    for b,(_,r) in zip(bars,x.iterrows()):
        ax.text(b.get_width()+.8,b.get_y()+b.get_height()/2,
            f'{r.no_show_rate:.1%}  |  n={int(r.completed_appointments):,}',va='center',fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/'appointment_type_no_show.png',dpi=180,bbox_inches='tight'); plt.close(fig)
outcome_chart(); lead_chart(); reminder_chart(); type_chart()
print(f'Rows: {n:,}; distinct patients: {patients:,}; completed: {done:,}')
print(f'No-show rate: {ns/done:.1%}; cancellation rate: {cancelled/n:.1%}')
print('Lead-time results:')
print(lead.to_string(index=False,formatters={'no_show_rate':lambda x:f'{x:.1%}'}))
print('Source SHA-256:',hashlib.sha256(SOURCE.read_bytes()).hexdigest())

