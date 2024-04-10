# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:04:07 2022

@author: hgathuri
"""

import requests
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
from io import StringIO
from datetime import date, timedelta
import matplotlib.pyplot as plt
from pylab import MaxNLocator

data = {
    'token': '',
    'content': 'record',
    'format': 'csv',
    'type': 'flat',
    'csvDelimiter': '',
    'rawOrLabel': 'raw',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'true',
    'returnFormat': 'csv'
}
r = requests.post('https://.........../',data=data)
print('HTTP Status: ' + str(r.status_code))

#Screening Log
sl_df = pd.read_csv(StringIO(r.text))

## Checking for duplicates
sl_df[sl_df[['uhid_ipno','participant_id','patient_name','age','sex']].duplicated()]

sl_df[sl_df[['uhid_ipno','patient_name','age','sex']].duplicated()].groupby(['redcap_data_access_group'])['hospital'].count()

#Dropping duplicates
sl_df2 = sl_df.drop_duplicates(subset=['uhid_ipno','patient_name','age','sex','participant_id'],keep='first')

#Dropping Embu
sl_df2 = sl_df2.loc[sl_df2['redcap_data_access_group']!='embu']

sl_df2['redcap_data_access_group'].value_counts()


#################################################################################
#SONIA TRIAL

data2 = {
    'token': '',
    'content': 'record',
    'format': 'csv',
    'type': 'flat',
    'csvDelimiter': '',
    'rawOrLabel': 'raw',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'true',
    'returnFormat': 'csv'
}
r = requests.post('https://................./',data=data2)
print('HTTP Status: ' + str(r.status_code))

st_df = pd.read_csv(StringIO(r.text),low_memory=False)
#st_df = pd.read_csv('C:\\Users\\hgathuri\\OneDrive - Kemri Wellcome Trust\\SONIA\DATA\\SONIA TRIAL\\SONIATrial_DATA_2022-05-07_2143.csv')

event2 = ['in_out_patient_d1',
'bp_sys_d1',
'bp_dia_d1',
'rand_blood_sugar_d1',
'temp_d1',
'saturation_d1',
'treatment_adherence_d1',
'new_diag_ae_d1',
'new_medication_d1',
'in_out_patient_d2',
'bp_sys_d2',
'bp_dia_d2',
'rand_blood_sugar_d2',
'temp_d2',
'saturation_d2',
'treatment_adherence_d2',
'new_diag_ae_d2',
'new_medication_d2',
'in_out_patient_d3',
'bp_sys_d3',
'bp_dia_d3',
'rand_blood_sugar_d3',
'temp_d3',
'saturation_d3',
'treatment_adherence_d3',
'new_diag_ae_d3',
'new_medication_d3',
'in_out_patient_d4',
'bp_sys_d4',
'bp_dia_d4',
'rand_blood_sugar_d4',
'temp_d4',
'saturation_d4',
'treatment_adherence_d4',
'new_diag_ae_d4',
'new_medication_d4',
'in_out_patient_d5',
'bp_sys_d5',
'bp_dia_d5',
'rand_blood_sugar_d5',
'temp_d5',
'saturation_d5',
'treatment_adherence_d5',
'new_diag_ae_d5',
'new_medication_d5',
'in_out_patient_d6',
'bp_sys_d6',
'bp_dia_d6',
'rand_blood_sugar_d6',
'temp_d6',
'saturation_d6',
'treatment_adherence_d6',
'new_diag_ae_d6',
'new_medication_d6',
'in_out_patient_d7',
'bp_sys_d7',
'bp_dia_d7',
'rand_blood_sugar_d7',
'temp_d7',
'saturation_d7',
'treatment_adherence_d7',
'new_diag_ae_d7',
'new_medication_d7',
'in_out_patient_d8',
'bp_sys_d8',
'bp_dia_d8',
'rand_blood_sugar_d8',
'temp_d8',
'saturation_d8',
'treatment_adherence_d8',
'new_diag_ae_d8',
'new_medication_d8',
'in_out_patient_d9',
'bp_sys_d9',
'bp_dia_d9',
'rand_blood_sugar_d9',
'temp_d9',
'saturation_d9',
'treatment_adherence_d9',
'new_diag_ae_d9',
'new_medication_d9',
'participant_withdraws',
'discharge_diagnosis',
'other_discharge_diagnosis',
'repeat_xray',
'repeat_xray_report',
'repeat_xray_image',
'respiratory_rate_discharge',
'heart_rate_discharge',
'temp_discharge',
'saturation_discharge',
'bp_sys_discharge',
'bp_dia_discharge',
'blood_sugar_discharge',
'date_of_discharge',
'date_of_death_hosp',
'cause_of_death_hosp',
'patient_alive_d14',
'date_of_death_d14',
'cause_of_death_d14',
'taking_medication_d14',
'days_took_prescription_d14',
'medication_not_taken_d14',
'new_illness',
'changed_contacts',
'patient_alive',
'date_of_death',
'cause_of_death',
'all_medication_taken',
'days_for_prescription',
'medication_not_taken',
'hospitalized_again',
'hospitalization_details',
'diagnosed_with_covid',
'covid_diagnosis_date',
'consultation_elsewhere',
'consultation_description',
'mobility',
'selfcare',
'usual_activities',
'pain_discomfort',
'anxiety_depression',
'other_relevant_information',
'final_status',
'date_of_withdrawal',
'last_contact_date',
'last_contact_details',
'termination_reason',
'other_termination_reason',
'withdrawal_reason',
'adverse_events',
'concomitant_medications',
]

event3 = ['adverse_event','ae_report_no',
'ae_details',
'clinic_visit_no',
'recent_steroid_dose',
'date_ae_reported',
'recent_steroid_date',
'initial_ae_report_date',
'ae_description',
'ae_start_date',
'ae_ongoing',
'ae_end_date',
'severity',
'steroid_related',
'expectedness',
'sae',
'outcome',
'action_ae',
'medication_prescription',
'medication_details',
'additional_comments']

# =============================================================================
# df1 = st_df.loc[st_df['redcap_event_name'] == "enrollment_and_bas_arm_1"]
# x1 = df1[df1.columns[~df1.columns.isin(event2)]]
# 
# df2 = st_df.loc[(st_df['redcap_event_name'] == "follow_up_and_ae_arm_1") & (st_df['redcap_repeat_instrument'].isnull())]
# event2.append('record_id')
# x2 = df2[df2.columns[df2.columns.isin(event2)]]
# event3.append('record_id')
# 
# df3 = st_df.loc[(st_df['redcap_event_name'] == "follow_up_and_ae_arm_1") & (st_df['redcap_repeat_instance']==1)]
# df33 = st_df.loc[(st_df['redcap_event_name'] == "follow_up_and_ae_arm_1") & (st_df['redcap_repeat_instance']==2)]
# 
# x3 =  df3[df3.columns[df3.columns.isin(event3)]]
# x33 =  df33[df33.columns[df33.columns.isin(event3)]]
# 
# dff = x1.merge(x2, on='record_id', how='left')
# dff21 = dff.merge(x3, on='record_id', how='left')
# dff2 = dff21.merge(x33, on='record_id', how='left')
# =============================================================================

############Reshaping dataset
df1 = st_df.loc[st_df['redcap_event_name'] == "enrollment_and_bas_arm_1"]
x1 = df1[df1.columns[~df1.columns.isin(event2)]]
df2 = st_df.loc[(st_df['redcap_event_name'] == "follow_up_and_ae_arm_1") & (st_df['redcap_repeat_instrument'].isnull())]
event2.append('record_id')
x2 = df2[df2.columns[df2.columns.isin(event2)]]

event3.extend(['record_id','redcap_repeat_instance'])

d3 = st_df.loc[st_df['redcap_repeat_instance'].notnull(),event3]

columns_values = list(set(d3.columns) - {'record_id', 'redcap_repeat_instance'})
df3 = d3.pivot(index='record_id', columns='redcap_repeat_instance', values=columns_values)

df3.columns = [i[0] + '__' + str(int(i[1]))for i in df3.columns]
dff = x1.merge(x2, on='record_id', how='left')

dff2 = dff.merge(df3, on='record_id', how='left')

###Omit KNH
#dff2 = dff2.loc[dff2['redcap_data_access_group']!='knh']


dff2['eligible'].value_counts(dropna=False)

### Total Recruited by end of last month
start_date = '2024-03-01'
end_date = '2024-03-31'


recruited_as_of_last_month = len(dff2[(dff2['randomization_number'].notnull()) & 
                                      (dff2['date_visit'] <= end_date)])

### Total recruited as of today
#total_recruited = len(dff2[dff2['randomization_number'].notnull()])


#In active follow-up by end of last month
active_followup = len(dff2.loc[(dff2['final_status'].isnull()) & 
                               (dff2['randomization_number'].notnull()) & 
                               (dff2['date_visit'] <= end_date)])

#Completed study by end of last month
completed_study = len(dff2.loc[((dff2['final_status']==5) | (dff2['final_status']==6) | 
                                (dff2['final_status']==7) & 
                                (dff2['date_visit'] <= end_date))])

#completed_study_df = dff2.loc[((dff2['final_status']==5) | (dff2['final_status']==6) & (dff2['date_visit'] <= '2022-09-30'))]
#completed_study_sc1 = len(completed_study_df.loc[completed_study_df['participant_rand_arm']==1])
#completed_study_str1 = len(completed_study_df.loc[completed_study_df['participant_rand_arm']==2])

# =============================================================================
# PREVIOUS MONTH
# =============================================================================

##Recruitment for previous month
recruited_lm = dff2.loc[dff2['date_visit'].between(start_date,end_date)]

recruited_lm_count = len(recruited_lm)


##Withdrawals for previous month
withdrawals_lm = len(dff2.loc[((dff2['final_status']==2) | (dff2['final_status']==3)) & 
                              (dff2['date_of_withdrawal'].between(start_date,end_date))])

##All withdrawals
total_withdrawals = len(dff2.loc[((dff2['final_status']==2) | (dff2['final_status']==3))])

#SAEs in last month

saes_lm = len(dff2.loc[(((dff2['sae__1'] !=9) & 
                         dff2['date_ae_reported__1'].between(start_date,end_date)) | 
                        ((dff2['sae__2'] != 9) & 
                         dff2['date_ae_reported__2'].between(start_date,end_date)) | 
                        ((dff2['sae__3'] != 9) & 
                         (dff2['date_ae_reported__3'].between(start_date,end_date)) | 
                         ((dff2['sae__4'] != 9) & 
                          (dff2['date_ae_reported__4'].between(start_date,end_date)))))])

dff2.loc[(((dff2['sae__1'] !=9) & dff2['date_ae_reported__1'].between(start_date,end_date)) |
         ((dff2['sae__2'] != 9) & dff2['date_ae_reported__2'].between(start_date,end_date)) |
         ((dff2['sae__3'] != 9) & (dff2['date_ae_reported__3'].between(start_date,end_date)) |
    ((dff2['sae__4'] != 9) & (dff2['date_ae_reported__4'].between(start_date,end_date))))),'participant_number'] 

##################
#Recruitment by Month Plots
###########################

### Set font type
csfont = {'fontname':'Times New Roman'}
     
lm_df = dff2.loc[dff2['date_visit'].between(start_date,end_date)]
#lm_df = dsmb_dff2

m = lm_df.groupby('redcap_data_access_group')['clinician_id'].count().reset_index()
m=m.rename(columns={'redcap_data_access_group': 'Hospital','clinician_id':'Patients Recruited'})

ax = m.set_index('Hospital').plot(kind='bar',figsize=(15,8))

#add overall title
ax.set_title("Number of Patients Recruited in March 2024", fontsize=28,**csfont)
#ax.set_title("Recruitment by Site", fontsize=24)
ax.set_xlabel('Hospitals',fontsize=22,**csfont)
ax.set_xticklabels(m.Hospital, fontsize=20, rotation=0,**csfont)
ax.set_ylabel('Count',fontsize=22,**csfont)
ya = ax.get_yaxis()
ya.set_major_locator(MaxNLocator(integer=True))
ax.get_legend().remove()
plt.yticks(fontsize=16,**csfont)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            '{:.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center',fontsize=22,**csfont,fontweight='bold')
    
plt.savefig('patients_recruited.png', dpi=400, bbox_inches="tight")
plt.show()

##Eligible and Consented

lm_sldf = sl_df2.loc[sl_df2['screening_date'].between(start_date,end_date)]
#lm_sldf = sl_df2 ## from DSMB

### Exclude sites that stopped recruiting
lm_sldf = lm_sldf.loc[(lm_sldf['redcap_data_access_group']=='bungoma') |
            (lm_sldf['redcap_data_access_group']=='machakos')|
            (lm_sldf['redcap_data_access_group']=='naivasha')|
            (lm_sldf['redcap_data_access_group']=='nakuru') |
            (lm_sldf['redcap_data_access_group']=='mbagathi')]

#eligible
el_lm_sldf = lm_sldf[lm_sldf['eligible']==1]
el_lm_sldf.groupby('redcap_data_access_group')['sex'].count().reset_index()

#consent
cnst_lm_sldf = el_lm_sldf[el_lm_sldf['consent']==1]
cnst_lm_sldf.groupby('redcap_data_access_group')['consent'].count().reset_index()

adms = lm_sldf.groupby('redcap_data_access_group')['eligible'].count().reset_index()
el = el_lm_sldf.groupby('redcap_data_access_group')['sex'].count().reset_index()
cnst = cnst_lm_sldf.groupby('redcap_data_access_group')['consent'].count().reset_index()
#cnst.loc[cnst['consent']==354,'consent'] =  353


# =============================================================================
# ## Get this from DSMB when doing overall plots
# Screened33 = Screened3.reset_index()
# Screened34 = Screened33.rename(columns={'index':'redcap_data_access_group'})
# eligible2 = eligible.reset_index()
# eligible3 = eligible2.rename(columns={'index':'redcap_data_access_group'})
# consented3 = consented2.reset_index()
# consented3 = consented3.rename(columns={'index':'redcap_data_access_group'})
# a2 = Screened34.merge(eligible3, how='outer', on='redcap_data_access_group')
# a3 = a2.merge(consented3, how='outer', on='redcap_data_access_group')
# a4 = a3.rename(columns={'redcap_data_access_group':'Hospital'})
# =============================================================================

a2 = adms.merge(el, how='outer', on='redcap_data_access_group')
a3 = a2.merge(cnst, how='outer', on='redcap_data_access_group')
a4 = a3.rename(columns={'redcap_data_access_group':'Hospital','eligible':'admissions','sex':'eligible','consent':'consented'})

a4 = a4.fillna(0)

###Imputing the missed records in SL
#lm_sldf.loc[(lm_sldf['eligible']==1) & (lm_sldf['consent']==1) & (lm_sldf['redcap_data_access_group']=='bungoma'),'participant_id']
a5 = a4.set_index('Hospital')
#a5.loc[(a5['consented']==14) & (a5['admissions']==116), 'consented'] = 13
#a5.loc['bungoma'] = a5.loc['bungoma'] + 2
#a5.loc['mbagathi'] = 0

##CONSENT RATE
#a5['Consent Rate'] = round(((a5['consented']/a5['eligible'])*100),2)


#a5.drop(['mama_lucy','kitale'], axis=0, inplace=True)

#########
#Plotting AEC
### Set font type
csfont = {'fontname':'Times New Roman'}

ax =  a5[['admissions', 'eligible', 'consented']].plot(kind='bar', fontsize=16,figsize=(15, 8))

ax =  a5.plot(kind='bar',figsize=(22, 12), width=0.8)

#add overall title and axis labels
ax.set_title("Number of Patients Admitted, Eligible and Consented",fontsize=24,**csfont)
ax.set_xlabel('Hospitals',fontsize=30,**csfont)
ax.set_xticklabels(a5.index, rotation=0,fontsize=30,**csfont)
#ax.set_xticks(rotation=60)
ax.set_ylabel('Count',fontsize=30,**csfont)
plt.title("Number of Patients Screened, Eligible and Consented in March 2024",fontsize=30,**csfont)
plt.yticks(fontsize=30,**csfont)

for q in ax.patches:
    width, height = q.get_width(), q.get_height()
    x, y = q.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            '{:.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center',fontsize=20,**csfont, fontweight='bold')


#plt.grid(True)
plt.savefig('AEC_plot.png',dpi=400,bbox_inches="tight")
plt.show()

#########################################################################
##ACTUAL AND PROJECTED Numbers
data = {'Months':['Mar 22','Apr 22','May 22','Jun 22','March 22','Aug 22','Sept 22','Oct 22','Nov 22','Dec 22',
                  'Jan 23','Feb 23','Mar 23','Apr 23','May 23','Jun 23','March 23','Aug 23','Sept 23'],
             'Actual recruitment': [0,8,48,141,184,124,92,88,134,71,105,88,109,71,105,143,112,np.nan,np.nan],
             'Previous Projection': [0, 181,181,181,181,181,181,181,181,181,181,181,189,112,np.nan,np.nan,np.nan,np.nan,np.nan],
             'Current Projection': [0, 121,121,121,121,121,121,121,121,121,121,121,121,121,121,121,121,121,123]}

### If we replace withdrawals
data = {'Months':['Mar 22','Apr 22','May 22','Jun 22','July 22','Aug 22','Sept 22','Oct 22','Nov 22','Dec 22',
                  'Jan 23','Feb 23','Mar 23','Apr 23','May 23','Jun 23','July 23','Aug 23','Sept 23','Oct 23',
                  'Nov 23','Dec 23','Jan 24','Feb 24','Mar 24','Apr 24'],
             'Actual recruitment': [0,8,48,141,184,124,92,88,134,71,105,88,109,71,105,143,112,125,92,59,
                                    31,14,39,52,np.nan,np.nan],
             'Previous Projection': [0, 181,181,181,181,181,181,181,181,181,181,181,189,112,np.nan,np.nan,np.nan,
                                     np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
             'Current Projection': [0, 102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,60,50,50,
                                    50,50]}

### Going by the initial target of 2180
data = {'Months':['Mar 22','Apr 22','May 22','Jun 22','July 22','Aug 22','Sept 22','Oct 22','Nov 22','Dec 22',
                  'Jan 23','Feb 23','Mar 23','Apr 23','May 23','Jun 23','July 23','Aug 23','Sept 23','Oct 23',
                  'Nov 23','Dec 23','Jan 24','Feb 24','Mar 24','Apr 24'],
             'Actual recruitment': [0,8,48,141,184,124,92,88,134,71,105,88,109,71,105,143,112,125,92,59,
                                    31,14,39,52,40,np.nan],
             'Previous Projection': [0, 181,181,181,181,181,181,181,181,181,181,181,189,112,np.nan,np.nan,np.nan,
                                     np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
             'Current Projection': [0, 102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,102,28,28,28,
                                    28,28]}


ap = pd.DataFrame(data)

ap['Previous Projection'] = ap['Previous Projection'].cumsum()
ap['Current Projection'] = ap['Current Projection'].cumsum()
ap['Actual recruitment'] = ap['Actual recruitment'].cumsum()


#ap.plot(x='Months')
#plt.show()



df_melt = pd.melt(ap, id_vars=['Months'], value_vars=['Actual recruitment','Previous Projection', 'Current Projection'])

#df_melt = df.melt(id_vars='Months', var_name='Months')
#df_melt = df_melt.fillna(0)
plt.figure(figsize = (20,15))
g=sns.relplot(
    kind='line',
    data=df_melt,
    x='Months', y='value',
    hue='variable',marker='o')


plt.ylim(0,2200)
#plt.grid(True)
plt.xticks(rotation=90,fontsize=12,**csfont)
plt.title("Projected versus Actual Recruitment",fontsize=20,**csfont)
plt.xlabel('Months',fontsize=16,**csfont)
plt.yticks(fontsize=12,**csfont)
##removes the title appearing in legend
g.legend.set_title(None)
plt.ylabel('Number of Participants',fontsize=16,**csfont)
g.map(plt.axhline, y=545, color=".7", dashes=(2, 1), zorder=0)
g.map(plt.axhline, y=1090, color=".7", dashes=(2, 1), zorder=0)
g.map(plt.axhline, y=1635, color=".7", dashes=(2, 1), zorder=0)
g.map(plt.axhline, y=2180, color=".8", dashes=(2, 1), zorder=0)
#plt.legend(loc='upper left')
plt.savefig('PA_plot2.png',bbox_inches = "tight")
#plt.savefig('PA_Aug.png',bbox_inches = "tight")
plt.show()

############################################################
####Plotting recruitment by month
######################################
# =============================================================================
# ap = ap.rename(columns={'Actual':'Count'}).drop('Projected',axis=1)
# ap.drop
# 
# ap2 = ap.loc[1:10].set_index('Months')
# 
# ax =  ap2.plot(kind='bar', fontsize=16,figsize=(20, 10))
# 
# #add overall title and axis labels
# ax.set_title("Recruitment by Month",fontsize=24)
# #ax.set_xlabel('Hospitals',fontsize=22)
# ax.set_xticklabels(ap2.index, rotation=60)
# #ax.set_xticks(rotation=60)
# #ax.set_ylabel('Number of Patients',fontsize=22)
# #.title("Number of Patients Admitted, Eligible and Consented in March 2024",fontsize=24)
# 
# 
# for q in ax.patches:
#     width, height = q.get_width(), q.get_height()
#     x, y = q.get_xy() 
#     ax.text(x+width/2, 
#             y+height/2, 
#             '{:.0f}'.format(height), 
#             horizontalalignment='center', 
#             verticalalignment='center')
# 
# 
# #plt.grid(True)
# #plt.savefig('recrutment_by_month.png',dpi=400,bbox_inches="tight")
# plt.show()
# 
# =============================================================================



#######################################
##### Cumulative vs Recruited by Previous month by Site
#########################################################

recruited_df = dff2[(dff2['randomization_number'].notnull()) & 
                    (dff2['date_visit'] <= end_date)]
cumulative_rec = recruited_df.\
    groupby('redcap_data_access_group')['participant_number'].count().reset_index()
#cumulative_rec['participant_number'].sum()


recruited_lm_df = dff2.loc[dff2['date_visit'].between(start_date,end_date)]
lm_rec = recruited_lm_df.groupby('redcap_data_access_group')['hosp_id'].count().reset_index()

cumulative_rec2 = cumulative_rec.merge(lm_rec, how='outer',
                                       on='redcap_data_access_group').fillna(0)
cumulative_rec3 = cumulative_rec2.\
    rename(columns={'redcap_data_access_group':'Hospital',
                    'participant_number':'Cumulative recruitment',
                    'hosp_id':'Recruited in March'})

## Filter recruiting sites
cumulative_rec4 = cumulative_rec3.loc[cumulative_rec3['Hospital'].isin(['bungoma','machakos','nakuru','naivasha','mbagathi'])]

a6 = cumulative_rec4.set_index('Hospital')

ax =  a6.plot(kind='bar',figsize=(22, 12))

#add overall title and axis labels
ax.set_title("Recruitment by Site - Cumulative vs Recruitment in March 2024",fontsize=28,**csfont)
ax.set_xlabel('Hospitals',fontsize=22,**csfont)
ax.set_xticklabels(a6.index, rotation=0,**csfont,fontsize=20)
plt.yticks(fontsize=20,**csfont)
#ax.set_xticks(rotation=60)
ax.set_ylabel('Number of Patients',fontsize=22,**csfont)
#.title("Number of Patients Admitted, Eligible and Consented in March 2024",fontsize=24)


for q in ax.patches:
    width, height = q.get_width(), q.get_height()
    x, y = q.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            '{:.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center',fontsize=16,**csfont,fontweight='bold')

#plt.grid(True)
plt.savefig('Cumulative_vs_recruitment.png',dpi=400,bbox_inches="tight")
plt.show()




#######################################################
###For Consort diagram
sl_df2_upto_lm = sl_df2.loc[sl_df2['screening_date'] <= end_date]

sl_df2_upto_lm_df = sl_df2.loc[sl_df2['screening_date'] <= end_date,['record_id','eligible','consent','participant_id']]



screened = len(sl_df2_upto_lm)
excluded_df = sl_df2_upto_lm.loc[sl_df2_upto_lm['participant_id'].isnull()]
excluded = len(excluded_df)

sl_df2_upto_lm['eligible'].value_counts(dropna=False)
sl_df2_upto_lm.loc[sl_df2_upto_lm['eligible'].isnull(),['redcap_data_access_group','participant_id','record_id']]

####Declined Consent
#sl_df2['screening_date'] = pd.to_datetime(sl_df2['screening_date'])

declined_consent_busia = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                             (excluded_df['screening_date'] >= '2022-04-18') & 
                                             (excluded_df['redcap_data_access_group'] == 'busia')])
declined_consent_bungoma = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] >= '2022-04-18') & 
                                               (excluded_df['redcap_data_access_group'] == 'bungoma')])
declined_consent_kisumu = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2022-04-19') & 
                                              (excluded_df['redcap_data_access_group'] == 'kisumu')])
declined_consent_naivasha = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                (excluded_df['screening_date'] >= '2022-04-21') & 
                                                (excluded_df['redcap_data_access_group'] == 'naivasha')])
declined_consent_kakamega = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                (excluded_df['screening_date'] >= '2022-04-27') & 
                                                (excluded_df['redcap_data_access_group'] == 'kakamega')])
declined_consent_machakos = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                (excluded_df['screening_date'] >= '2022-04-26') & 
                                                (excluded_df['redcap_data_access_group'] == 'machakos')])
declined_consent_cgtrh = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                             (excluded_df['screening_date'] >= '2022-05-05') & 
                                             (excluded_df['redcap_data_access_group'] == 'cgtrh')])
declined_consent_kilifi = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2022-05-09') & 
                                              (excluded_df['redcap_data_access_group'] == 'kilifi')])
declined_consent_mbagathi = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                (excluded_df['screening_date'] >= '2022-05-09') & 
                                                (excluded_df['redcap_data_access_group'] == 'mbagathi')])
declined_consent_kutrh = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                             (excluded_df['screening_date'] >= '2022-07-04') & 
                                             (excluded_df['redcap_data_access_group'] == 'kutrh')])
declined_consent_mama_lucy = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                 (excluded_df['screening_date'] >= '2022-08-15') & 
                                                 (excluded_df['redcap_data_access_group'] == 'mama_lucy')])
declined_consent_kitale = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2022-08-16') & 
                                              (excluded_df['redcap_data_access_group'] == 'kitale')])
declined_consent_kiambu = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2022-05-12') & 
                                              (excluded_df['redcap_data_access_group'] == 'kiambu')])
declined_consent_nakuru = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2023-06-02') & 
                                              (excluded_df['redcap_data_access_group'] == 'nakuru')])
declined_consent_knh = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] >= '2023-06-02') & 
                                              (excluded_df['redcap_data_access_group'] == 'knh')])





declined_consent = declined_consent_busia+declined_consent_bungoma+declined_consent_kisumu+declined_consent_naivasha+\
                    declined_consent_kakamega+declined_consent_machakos+declined_consent_cgtrh+declined_consent_kilifi+\
                    declined_consent_mbagathi+declined_consent_kutrh+declined_consent_mama_lucy+declined_consent_kitale+\
                    declined_consent_kiambu+declined_consent_nakuru+declined_consent_knh

####before SIV hence not eligible
declined_consent_busia2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] < '2022-04-18') & 
                                              (excluded_df['redcap_data_access_group'] == 'busia')])
declined_consent_bungoma2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                (excluded_df['screening_date'] < '2022-04-18') & 
                                                (excluded_df['redcap_data_access_group'] == 'bungoma')])
declined_consent_kisumu2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2022-04-19') & 
                                               (excluded_df['redcap_data_access_group'] == 'kisumu')])
declined_consent_naivasha2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                 (excluded_df['screening_date'] < '2022-04-21') & 
                                                 (excluded_df['redcap_data_access_group'] == 'naivasha')])
declined_consent_kakamega2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                 (excluded_df['screening_date'] < '2022-04-27') & 
                                                 (excluded_df['redcap_data_access_group'] == 'kakamega')])
declined_consent_machakos2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                 (excluded_df['screening_date'] < '2022-04-26') & 
                                                 (excluded_df['redcap_data_access_group'] == 'machakos')])
declined_consent_cgtrh2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] < '2022-05-05') & 
                                              (excluded_df['redcap_data_access_group'] == 'cgtrh')])
declined_consent_kilifi2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2022-05-09') & 
                                               (excluded_df['redcap_data_access_group'] == 'kilifi')])
declined_consent_mbagathi2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                 (excluded_df['screening_date'] < '2022-05-09') & 
                                                 (excluded_df['redcap_data_access_group'] == 'mbagathi')])
declined_consent_kutrh2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                              (excluded_df['screening_date'] < '2022-07-04') & 
                                              (excluded_df['redcap_data_access_group'] == 'kutrh')])
declined_consent_mama_lucy2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                                  (excluded_df['screening_date'] < '2022-08-15') & 
                                                  (excluded_df['redcap_data_access_group'] == 'mama_lucy')])
declined_consent_kitale2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2022-08-16') & 
                                               (excluded_df['redcap_data_access_group'] == 'kitale')])
declined_consent_kiambu2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2022-05-12') & 
                                               (excluded_df['redcap_data_access_group'] == 'kiambu')])
declined_consent_nakuru2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2023-06-02') & 
                                               (excluded_df['redcap_data_access_group'] == 'nakuru')])
declined_consent_knh2 = len(excluded_df.loc[(excluded_df['eligible']==1) & (excluded_df['consent']==0) & 
                                               (excluded_df['screening_date'] < '2023-06-02') & 
                                               (excluded_df['redcap_data_access_group'] == 'knh')])



not_eligible1 = declined_consent_busia2+declined_consent_bungoma2+declined_consent_kisumu2+declined_consent_naivasha2+\
                declined_consent_kakamega2+declined_consent_machakos2+declined_consent_cgtrh2+declined_consent_kilifi2+\
                    declined_consent_mbagathi2+declined_consent_kutrh2+declined_consent_mama_lucy2+declined_consent_kitale2+\
                        declined_consent_kiambu+declined_consent_nakuru2+declined_consent_knh2
                        
not_eligible2 = len(excluded_df.loc[excluded_df['eligible']==0]) 
not_eligible = not_eligible2 + not_eligible1



##Randomized and completed/withdrawn/ltf
randomized_df = recruited_df.loc[recruited_df['final_status'].notnull()]

##DF to merge with sl_df2_upto_lm_df
randomized_df2 = randomized_df[['participant_number','participant_rand_arm','final_status','termination_reason','other_termination_reason']]

randomized = len(randomized_df)
##Arms
sc_df = randomized_df.loc[randomized_df['participant_rand_arm']==1]
sc = len(sc_df)
strr_df = randomized_df.loc[randomized_df['participant_rand_arm']==2]
strr = len(strr_df)

#Withdrawals by Arm
sc_withdrawal_df = sc_df.loc[(sc_df['final_status']==2) | (sc_df['final_status']==3)]
sc_withdrawal = len(sc_withdrawal_df)
withdrew_consent_sc = len(sc_df.loc[sc_df['termination_reason']==2])
sc_withdrawal_df['termination_reason'].value_counts(dropna=False)
#sc_withdrawal_df.loc[sc_withdrawal_df['termination_reason']==6,'other_termination_reason'].to_csv('SC_Withdrawals.csv')


strr_withdrawal_df = strr_df.loc[(strr_df['final_status']==2) | (strr_df['final_status']==3)]
strr_withdrawal = len(strr_withdrawal_df)
withdrew_consent_strr = len(strr_df.loc[strr_df['termination_reason']==2])
strr_withdrawal_df['termination_reason'].value_counts(dropna=False)
#strr_withdrawal_df.loc[strr_withdrawal_df['termination_reason']==6,'other_termination_reason'].to_csv('Str_Withdrawals.csv')

strr_withdrawal_df.loc[strr_withdrawal_df['termination_reason'].isnull(),'record_id']


##Lost to follow-up by Arms
sc_ltf = len(sc_df.loc[sc_df['final_status']==4])
strr_ltf = len(strr_df.loc[strr_df['final_status']==4])

##Completed Study by Arms
completed_sc = len(sc_df.loc[(sc_df['final_status']==5) | 
                             (sc_df['final_status']==6)])
completed_sc = len(sc_df.loc[(sc_df['final_status']==5) | 
                            (sc_df['final_status']==6) | 
                             (sc_df['final_status']==7)])
completed_str = len(strr_df.loc[(strr_df['final_status']==5) | 
                                (strr_df['final_status']==6)])

completed_str = len(strr_df.loc[(strr_df['final_status']==5) | 
                                (strr_df['final_status']==6) |
                                (strr_df['final_status']==7)])

#completed_study_sc = len(sc_df.loc[((sc_df['final_status']==5) | (sc_df['final_status']==6) & (sc_df['date_visit'] <= '2022-12-31'))])

sc_df['diagnosed_with_covid'].value_counts(dropna=False)
strr_df['diagnosed_with_covid'].value_counts(dropna=False)




###Steroids
strr_df = dff2.loc[dff2['participant_rand_arm']==2]
strr_df['final_status'].value_counts(dropna=False)
strr_df['termination_reason'].value_counts(dropna=False)

strr_df.loc[strr_df['final_status']==4,['redcap_data_access_group','participant_number']]

completed_study_strr = len(strr_df.loc[((strr_df['final_status']==5) | (strr_df['final_status']==6) & (strr_df['date_visit'] <= '2022-09-30'))])

##Merging SL and STRL
sl_df2_upto_lm_df.columns
randomized_df2.columns
randomized_df2.rename(columns={'participant_number':'participant_id'},inplace=True)

##merging the two
consort_df = pd.merge(sl_df2_upto_lm_df,randomized_df2,on='participant_id',how='outer')
consort_df['exc'] = np.nan
consort_df.loc[consort_df['eligible']==0,'exc'] = 'Not eligible'
consort_df.loc[(consort_df['eligible']==1) & (consort_df['consent']==0),'exc'] = 'Declined consent'
consort_df.loc[consort_df['exc'].isnull(),['eligible','consent']]

consort_df['ltfu'] = np.nan
consort_df.loc[consort_df['final_status']==4,'ltfu'] = 'Lost to follow up'

##Completed study

consort_df['complete'] = np.nan
consort_df.loc[(consort_df['final_status']==5) & (consort_df['final_status']==6),'complete'] = 'Completed Study'
consort_df.to_csv('consort_df.csv',index=False)
