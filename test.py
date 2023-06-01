import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMClassifier
import scipy.stats as stats
#%%
data = pd.read_csv('ethusdt_20220301_0630_20s_trades.csv')
#%%
data['datetime'] = pd.to_datetime(data['timestamp']+28800, unit='s')
#%%
data.to_csv('ethusdt_20220301_0630_20s_trades.csv')
#%%
data = pd.read_csv('ethusdt_20220601_0630_20s_wap1_strategy_8.0_20220909.csv')
#%%
data['datetime'] = pd.to_datetime(data['datetime'])
data['timestamp'] = data.datetime.values.astype(np.int64)//10**9-28800
#%%
data = data.iloc[:,1:]
#%%
print(data['predict'].describe())
#%%
df_1 = data.loc[data['predict']>0.6]
df_0 = data.loc[data['predict']<0.4]
#%%
time60 = []
num = 0
for i, time in enumerate(df_1['timestamp']):
    # print(time)
    dt60 = data[data['timestamp'] == time + 60].values
    if len(dt60) > 0:
        time60.append(dt60[0][2])
    else:
        time60.append(0)
df_1['wap1_1min'] = time60
print(df_1)
#%%
time60 = []
num = 0
for i, time in enumerate(df_0['timestamp']):
    # print(time)
    dt60 = data[data['timestamp'] == time + 60].values
    if len(dt60) > 0:
        time60.append(dt60[0][2])
    else:
        time60.append(0)
df_0['wap1_1min'] = time60
print(df_0)
#%%
df_1['log'] = np.log(df_1.wap1_1min/df_1.wap1)
print(abs(df_1['log']).describe())
df_0['log'] = np.log(df_0.wap1_1min/df_0.wap1)
print(abs(df_0['log']).describe())
#%%
df_1.to_csv('ethusdt_20220401_0430_single1.csv')
df_0.to_csv('ethusdt_20220401_0430_single0.csv')
#%%
test_1= len(df_1[(df_1.predict>0.65)&(df_1.target>0)])/len(df_1)
df_0['target'] = np.where(df_0['target']>0,1,-1)
test_0= len(df_0[(df_0.predict<0.35)&(df_0.target<0)])/len(df_0)
print(test_1)
print(test_0)
#%%
data = data.set_index('datetime')
#%%
test = data[data.index<='2022-06-15']
#%%
df_1_0615 = test.loc[test['predict']>0.6]
df_0_0615 = test.loc[test['predict']<0.4]
#%%
test_1_0615 = len(df_1_0615[(df_1_0615.predict>0.6)&(df_1_0615.target>0)])/len(df_1_0615)
df_0_0615['target'] = np.where(df_0_0615['target']>0,1,-1)
test_0_0615 = len(df_0_0615[(df_0_0615.predict<0.4)&(df_0_0615.target<0)])/len(df_0_0615)
print(test_1_0615)
print(test_0_0615)
#%%
data['Predict'] = np.where(data['Predict']>0.496581,1,-1)
data['target'] = np.where(data['target']>0,1,-1)

#%%
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime').resample('1min').apply('last')
#%%
test = len(data[(data.Predict>0)&(data.target>0)|((data.Predict<0)&(data.target<0))])/len(data)
print(test)
#%%
stats.jarque_bera(data['target'])
#%%
data.to_csv('btcusdt_20220127_0131_5min_last_price_strategy_1.0_20220802_test.csv')


