from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
# month = 1
all_data = pd.DataFrame()
for month in tqdm(range(1, 8)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol) and ('year', '=', year) and ('month','=',month)]
    dataset_base = pq.ParquetDataset('/songhe/AI1.0/data/feat/trade_1s_base/gate_swap_u', filters=filters)
    trade_base = dataset_base.read_pandas().to_pandas()
    trade_base = trade_base.iloc[:, :-3]
    dataset_vwap = pq.ParquetDataset('/songhe/AI1.0/data/feat/trade_1s_vwap/gate_swap_u', filters=filters)
    trade_vwap = dataset_vwap.read_pandas().to_pandas()
    trade_vwap = trade_vwap.iloc[:, :-3]
    dataset_price = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_price/gate_swap_u',filters=filters)
    depth_price = dataset_price.read_pandas().to_pandas()
    depth_price = depth_price.iloc[:, :-3]
    dataset_size = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_size/gate_swap_u', filters=filters)
    depth_size = dataset_size.read_pandas().to_pandas()
    depth_size = depth_size.iloc[:, :-3]
    dataset_wap1 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap1/gate_swap_u', filters=filters)
    depth_wap1 = dataset_wap1.read_pandas().to_pandas()
    depth_wap1 = depth_wap1.iloc[:, :-3]
    dataset_wap2 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap2/gate_swap_u', filters=filters)
    depth_wap2 = dataset_wap2.read_pandas().to_pandas()
    depth_wap2 = depth_wap2.iloc[:, :-3]
    dataset_wap3 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap3/gate_swap_u', filters=filters)
    depth_wap3 = dataset_wap3.read_pandas().to_pandas()
    depth_wap3 = depth_wap3.iloc[:, :-3]
    dataset_wap4 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap4/gate_swap_u', filters=filters)
    depth_wap4 = dataset_wap4.read_pandas().to_pandas()
    depth_wap4 = depth_wap4.iloc[:, :-3]
    dataset_wap5 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap5/gate_swap_u', filters=filters)
    depth_wap5 = dataset_wap5.read_pandas().to_pandas()
    depth_wap5 = depth_wap5.iloc[:, :-3]
    list_of_datasets = [trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5]
    data_merge = reduce(
        lambda left, right: pd.merge(left, right, on=['closetime'], how='inner', suffixes=['', "_drop"]),
        list_of_datasets)
    data_merge.drop([col for col in data_merge.columns if 'drop' in col], axis=1, inplace=True)
    all_data = all_data.append(data_merge)

#%%
del trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5, list_of_datasets, data_merge
del dataset_wap5, dataset_wap3, dataset_wap4, dataset_wap2, dataset_wap1, dataset_base, dataset_vwap, dataset_price, dataset_size
#%%
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800, unit='s')
#%%
data = all_data[(all_data.datetime>='2022-01-01 08:00:01')&(all_data.datetime<='2022-03-03 23:59:59')]
#%%
del all_data
#%%
data = data.drop(['closetime'], axis=1)
#%%
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist
#%%
# data = time_group_trade
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d

forceindex = ForceIndexIndicator(close=data['last_price'], volume=data['size'])
data['forceindex'] = forceindex.force_index()
# easyofmove = EaseOfMovementIndicator(high=data['last_price_amax'], low=data['last_price_amin'], volume=data['volume_size_y'])
# data['easyofmove'] = easyofmove.ease_of_movement()
# bollingband = BollingerBands(close=data['last_price'])
# data['bollingerhband'] = bollingband.bollinger_hband()
# data['bollingerlband'] = bollingband.bollinger_lband()
# data['bollingermband'] = bollingband.bollinger_mavg()
# data['bollingerpband'] = bollingband.bollinger_pband()
# data['bollingerwband'] = bollingband.bollinger_wband()
# keltnerchannel = KeltnerChannel(high=data['last_price_amax'], low=data['last_price_amin'], close=data['last_price'])
# data['keltnerhband'] = keltnerchannel.keltner_channel_hband()
# data['keltnerlband'] = keltnerchannel.keltner_channel_lband()
# data['keltnerwband'] = keltnerchannel.keltner_channel_wband()
# data['keltnerpband'] = keltnerchannel.keltner_channel_pband()
# donchichannel = DonchianChannel(high=data['last_price_amax'], low=data['last_price_amin'], close=data['last_price'])
# data['donchimband'] = donchichannel.donchian_channel_mband()
# data['donchilband'] = donchichannel.donchian_channel_lband()
# data['donchipband'] = donchichannel.donchian_channel_pband()
# data['donchiwband'] = donchichannel.donchian_channel_wband()
macd = MACD(close=data['last_price'])
data['macd'] = macd.macd()
data['macdsignal'] = macd_signal(close=data['last_price'])
data['macddiff'] = macd_diff(close=data['last_price'])
smafast = SMAIndicator(close=data['last_price'],window=16)
data['smafast'] = smafast.sma_indicator()
smaslow = SMAIndicator(close=data['last_price'],window=32)
data['smaslow'] = smaslow.sma_indicator()
data['stochrsi'] = stochrsi(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_k'] = stochrsi_k(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_d'] = stochrsi_d(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')
data = data.replace(np.inf, 1)
data = data.replace(-np.inf, -1)
#%%
data = data.set_index('datetime')
#%%
data = reduce_mem_usage(data)[0]
#%%
data['target'] = np.log(data['last_price']/data['last_price'].shift(60))*100
data['target'] = data['target'].shift(-60)
#%%
data = data.dropna(axis=0, how='any')
#%%
def classify(y):

    if y < 0:
        return 0
    if y > 0:
        return 1
    else:
        return -1
data['target'] = data['target'].apply(lambda x:classify(x))
print(data['target'].value_counts())
#%%
data = data[~data['target'].isin([-1])]
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[train.index < '2022-02-26 00:00:00']
# train_set = train[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-31 23:59:59')]
test_set = train[(train.index >= '2022-02-26 00:00:00')&(train.index <= '2022-03-02 23:59:59')]
train_target = target[train.index < '2022-02-26 00:00:00']
# train_target = target[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-31 23:59:59')]
test_target = target[(train.index >= '2022-02-26 00:00:00')&(train.index <= '2022-03-02 23:59:59')]
#%%
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)
#%%
model = CatBoostClassifier(iterations=20,
                            learning_rate=0.1,
                            depth=5,
                            loss_function='Logloss',
                            silent=True)
#%%
model.fit(X_train, X_train_target)
y_pred = model.predict_proba(X_test)[:, 1]
#%%
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from numpy import sqrt
from numpy import argmax
fpr, tpr, thresholds = roc_curve(X_test_target,y_pred)
gmeans = sqrt(tpr * (1 - fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#%%
yhat = [1 if y > thresholds[ix] else 0 for y in y_pred]
print("训练集表现：")
print(classification_report(X_test_target, yhat))
print("AUC：", roc_auc_score(X_test_target, yhat))
#%%
test_data = data.reset_index()
predict = pd.DataFrame(y_pred, columns=['Predict'])
predict['datetime'] = test_data['datetime']
predict['last_price'] = test_data['last_price']
predict['target'] = test_data['target']
# predict.to_csv('btcusdt_20220226_0302_1min_last_price_strategy_1.0_20220805_catboost.csv')
#%%
df_1 = predict.loc[predict['Predict']>0.7]
df_0 = predict.loc[predict['Predict']<0.3]
test_1 = len(df_1[(df_1.Predict>0.7)&(df_1.target>0)])/len(df_1)
df_0['target'] = np.where(df_0['target']>0,1,-1)
test_0 = len(df_0[(df_0.Predict<0.7)&(df_0.target<0)])/len(df_0)
print(test_1)
print(test_0)