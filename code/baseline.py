#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn.model_selection  import train_test_split,cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score,roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

import lightgbm as lgb

import warnings

warnings.simplefilter('ignore')


# In[2]:


train = pd.read_csv('../data/train_dataset.csv', delimiter='\t')
test =  pd.read_csv('../data/test_dataset.csv', delimiter='\t')


# In[3]:


data = pd.concat([train, test])
print(data.shape)


# In[4]:


data['browser'] = data['browser_source'].astype(str) + '_' +  data['browser_type'].astype(str)+ '_'                 + data['bus_system_code'].astype(str)+ '_'+ data['os_type'].astype(str) + '_'                + data['os_version'].astype(str) + '_' + data['device_model'].astype(str) + '_'                 + data['ip'].astype(str) + '_' +  data['ip_location_type_keyword'].astype(str)+ '_'                 + data['ip_risk_level'].astype(str) 
                

data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
data['auth_type'].fillna('__NaN__', inplace=True)  



for col in tqdm(['user_name', 'action', 'ip', 'auth_type','browser',
                 'ip_location_type_keyword', 'ip_risk_level', 'location', 'device_model',
                 'os_type', 'os_version', 'browser_type', 'browser_version',
                 'bus_system_code', 'op_target']):
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])


# In[5]:


# 时间特征
data['op_date'] = pd.to_datetime(data['op_date'])
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
data['ts_diff1'] = data['op_ts'] - data['last_ts']

for method in ['mean', 'max', 'min', 'std', 'sum', 'median', 'skew']:
    for col in ['user_name', 'ip', 'location', 'device_model', 'os_version', 'browser_version','browser']:
        data[f'ts_diff1_{method}_' + str(col)] = data.groupby(col)['ts_diff1'].transform(method)


# In[7]:


time_columns = ['is_month_end', 'is_month_start',                'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']

data['is_leap_year'] = data['op_date'].dt.is_leap_year
data['is_month_end'] = data['op_date'].dt.is_month_end
data['is_month_start'] = data['op_date'].dt.is_month_start
data['is_quarter_end'] = data['op_date'].dt.is_quarter_end
data['is_quarter_start'] = data['op_date'].dt.is_quarter_start
data['is_year_end'] = data['op_date'].dt.is_year_end
data['is_year_start'] = data['op_date'].dt.is_year_start


for col in tqdm(time_columns):
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])


# In[8]:


# 众数和分位数
for method in ['nunique']:
    for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
        data[f'user_{f}_{method}'] = data.groupby(['user_name'])[f].transform(method)
    
    


# In[10]:



train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]

print(train.shape, test.shape)


# In[11]:


# 训练
ycol = 'risk_label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'session_id', 'op_date', 'last_ts', 'last_ts2', 'precise_time'], train.columns))




model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='goss',
                           tree_learner='serial',
                           num_leaves=2 ** 8,
                           max_depth=16,
                           learning_rate=0.2,
                           n_estimators=10000,
                           subsample=0.75,
                           feature_fraction=0.55,
                           reg_alpha=0.2,
                           reg_lambda=0.2,
                           random_state=1983,
                           is_unbalance=True,
                           # scale_pos_weight=130,
                           metric='auc')


oof = []
prediction = test[['session_id']]
prediction[ycol] = 0
df_importance_list = []

kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=1983)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['session_id', ycol]].copy()
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)

    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction[ycol] += pred_test[:, 1] / kfold.n_splits

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)


df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
df_importance


# In[12]:


df_oof = pd.concat(oof)
print('roc_auc_score', roc_auc_score(df_oof[ycol], df_oof['pred']))


# In[13]:


prediction['id'] = range(len(prediction))
prediction['id'] = prediction['id'] + 1
prediction = prediction[['id', 'risk_label']].copy()
prediction.columns = ['id', 'ret']
prediction.head()


# In[14]:


prediction['ret'].describe()


# In[15]:


# prediction.to_csv('../sub/1113baseline.csv', index=False)


# In[16]:


# 绘制roc曲线
# df_oof
tpr = []
fpr = []

def roc(real, pred):
#     for i in tqdm(range(0, 101)):
    for i in range(0, 101):
        prob = i/100.0
        tmp = df_oof[df_oof['risk_label']==1]['pred'].apply(lambda x:x>=prob)
        tmp = pd.DataFrame(tmp).reset_index(drop=True)
        tp = len(tmp[tmp['pred']==True])
        
        tmp = df_oof[df_oof['risk_label']==0]['pred'].apply(lambda x:x<prob)
        tmp = pd.DataFrame(tmp).reset_index(drop=True)
        tn = len(tmp[tmp['pred']==True])
        
        tmp = df_oof[df_oof['risk_label']==0]['pred'].apply(lambda x:x>=prob)
        tmp = pd.DataFrame(tmp).reset_index(drop=True)
        fp = len(tmp[tmp['pred']==True])
        
        tmp = df_oof[df_oof['risk_label']==1]['pred'].apply(lambda x:x<prob)
        tmp = pd.DataFrame(tmp).reset_index(drop=True)
        fn = len(tmp[tmp['pred']==True])
        
#         print(f'tp is{tp}, tn is {tn}, fp is {fp}, fn is {fn}')
        
        tmp_tpr = tp/(tp+fn)
        tmp_fpr = fp/(tn+fp)
        tpr.append(tmp_tpr)
        fpr.append(tmp_fpr)

        
roc(df_oof['risk_label'], df_oof['pred'])
        
        
    
    


# In[17]:


import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.show()


# In[18]:


# auc计算的实现
df = df_oof.copy()
M = 2940
N = 12076
tmp = df.sort_values('pred').reset_index(drop=True)
t = tmp[tmp['risk_label']==1.0]
# 计算auc的值
t = t.reset_index()
t['index'] = t['index'] + 1
auc = (t['index'].sum()-M*(M+1)/2)/(M*N)
print(f'train auc is: {auc}')


# In[ ]:




