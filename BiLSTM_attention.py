# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import tensorflow as tf

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Attention, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.clear_session()
import time
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
# month = 1
for month in range(1, 2):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol) and ('year', '=', year) and ('month','=',month)]
    dataset_base = pq.ParquetDataset('feat/trade_1s_base/gate_swap_u', filters=filters)
    trade_base = dataset_base.read_pandas().to_pandas()
    trade_base = trade_base.iloc[:, :-3]
    dataset_vwap = pq.ParquetDataset('feat/trade_1s_vwap/gate_swap_u', filters=filters)
    trade_vwap = dataset_vwap.read_pandas().to_pandas()
    trade_vwap = trade_vwap.iloc[:, :-3]
    dataset_price = pq.ParquetDataset('feat/depth_1s_price/gate_swap_u',filters=filters)
    depth_price = dataset_price.read_pandas().to_pandas()
    depth_price = depth_price.iloc[:, :-3]
    dataset_size = pq.ParquetDataset('feat/depth_1s_size/gate_swap_u', filters=filters)
    depth_size = dataset_size.read_pandas().to_pandas()
    depth_size = depth_size.iloc[:, :-3]
    dataset_wap1 = pq.ParquetDataset('feat/depth_1s_wap1/gate_swap_u', filters=filters)
    depth_wap1 = dataset_wap1.read_pandas().to_pandas()
    depth_wap1 = depth_wap1.iloc[:, :-3]
    dataset_wap2 = pq.ParquetDataset('feat/depth_1s_wap2/gate_swap_u', filters=filters)
    depth_wap2 = dataset_wap2.read_pandas().to_pandas()
    depth_wap2 = depth_wap2.iloc[:, :-3]
    dataset_wap3 = pq.ParquetDataset('feat/depth_1s_wap3/gate_swap_u', filters=filters)
    depth_wap3 = dataset_wap3.read_pandas().to_pandas()
    depth_wap3 = depth_wap3.iloc[:, :-3]
    dataset_wap4 = pq.ParquetDataset('feat/depth_1s_wap4/gate_swap_u', filters=filters)
    depth_wap4 = dataset_wap4.read_pandas().to_pandas()
    depth_wap4 = depth_wap4.iloc[:, :-3]
    dataset_wap5 = pq.ParquetDataset('feat/depth_1s_wap5/gate_swap_u', filters=filters)
    depth_wap5 = dataset_wap5.read_pandas().to_pandas()
    depth_wap5 = depth_wap5.iloc[:, :-3]
    list_of_datasets = [trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5]
# #%%
# for index, dataset in enumerate(list_of_datasets):
#     # first = list_of_datasets[0].columns
#     # second = list_of_datasets[1].columns
#     # common = list(set(first)&set(second))
#     # print(common)
#     # if common == ['closetime']:
#     # data_merge = pd.merge(list_of_datasets[0], list_of_datasets[1], on='closetime', how='left')
#     # if index > 1:
#     a = list_of_datasets[0].columns
#     if index > 0:
#         b = list_of_datasets[index].columns
#         common_a_b = list(set(a)&set(b))
#         # print(common_a_b)
#         data_merge = pd.merge(list_of_datasets[0],list_of_datasets[index], on='closetime', how='left')
#         if common_a_b == ['closetime']:
#             data_merge = pd.merge(data_merge, list_of_datasets[index], on='closetime', how='left')
#             print(data_merge)
#             print('no common')
#         else:
#             common_a_b.remove('closetime')
#             list_of_datasets[index] = list_of_datasets[index].drop(common_a_b, axis=1)
#             data_merge = pd.merge(data_merge, list_of_datasets[index], on='closetime', how='left')
#             print('have common')
#             print(data_merge)

#%%
from functools import reduce

data_merge = reduce(lambda left, right:pd.merge(left, right, on=['closetime'], how='inner', suffixes=['', "_drop"]), list_of_datasets)
data_merge.drop([col for col in data_merge.columns if 'drop' in col], axis=1, inplace=True)
#%%
del trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5, list_of_datasets
#%%
data_merge['datetime'] = pd.to_datetime(data_merge['closetime']+28800, unit='s')
#%%
data = data_merge[data_merge.datetime>='2022-01-01 08:00:01']
#%%
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')
data = data.replace(np.inf, 1)
data = data.replace(-np.inf, -1)
#%%
data = data.drop(['closetime'], axis=1)
#%%
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


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

forceindex = ForceIndexIndicator(close=data['price'], volume=data['size'])
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
macd = MACD(close=data['price'])
data['macd'] = macd.macd()
data['macdsignal'] = macd_signal(close=data['price'])
data['macddiff'] = macd_diff(close=data['price'])
smafast = SMAIndicator(close=data['price'],window=16)
data['smafast'] = smafast.sma_indicator()
smaslow = SMAIndicator(close=data['price'],window=32)
data['smaslow'] = smaslow.sma_indicator()
data['stochrsi'] = stochrsi(close=data['price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_k'] = stochrsi_k(close=data['price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_d'] = stochrsi_d(close=data['price'],window=9, smooth1=26, smooth2=12)
data = data.fillna(method='bfill')
data = data.replace(np.inf, 1)
data = data.replace(-np.inf, -1)

#%%
data = data.set_index('datetime')
#%%
data = reduce_mem_usage(data)[0]
#%%
data['target'] = np.log(data['price']/data['price'].shift(60))*100
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
cols = data.columns  # 所有列
train_col = []  # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"]  # 取前26列为训练数据，最后一列为target
#%%
train_set = train[train.index < '2022-01-05 00:00:00']
# train_set = train[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-31 23:59:59')]
test_set = train[(train.index >= '2022-01-05 00:00:00')&(train.index <= '2022-01-06 23:59:59')]
train_target = target[train.index < '2022-01-05 00:00:00']
# train_target = target[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-31 23:59:59')]
test_target = target[(train.index >= '2022-01-05 00:00:00')&(train.index <= '2022-01-06 23:59:59')]
#%%
# 将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)  # 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

from keras.utils.np_utils import to_categorical

train_target = to_categorical(train_target, num_classes=2)
test_target = to_categorical(test_target, num_classes=2)
#%%
# from numpy import array
# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
def data_split(sequence, target, n_timestamp):
    X = []
    y = []
    X_target = []
    y_target = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        seq_target_x, seq_target_y = target[i:end_ix], target[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        X_target.append(seq_target_x)
        y_target.append(seq_target_y)

    return array(X), array(y), array(X_target), array(y_target)

#%%
n_timestamp = 30

X_train, y_train, X_train_target, y_train_target = data_split(train_set_scaled, train_target, n_timestamp)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
X_test, y_test, X_test_target, y_test_target = data_split(test_set_scaled, test_target, n_timestamp)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
#%%
# cols = data.columns  # 所有列
# train_col = []  # 选择测试集的列
# for i in cols:
#     if i != "target":
#         train_col.append(i)
#
# train = data[train_col]
# target = data["target"]  # 取前26列为训练数据，最后一列为target
#
# # 将数据归一化，范围是0到1
# from sklearn.preprocessing import MinMaxScaler
# from keras.utils.np_utils import to_categorical
# from sklearn.model_selection import train_test_split
#
# sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
# train = sc.fit_transform(train)  # 数据归一
# target = np.array(target)
# target = to_categorical(target, num_classes=3)
#
# train_X, test_X, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=3)
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

#%%
from tensorflow.keras.models import Model
class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def my_model():
    inputs = Input(shape=(394, 1))

    # BiLSTM层
    x = Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(245, 1))(inputs)
    x = Dropout(0.5)(x)

    # Attention层
    x = AttentionLayer(attention_size=100)(x)

    # BiLSTM层
    x = Bidirectional(LSTM(units=50), input_shape=(49, 1))(inputs)
    x = Dropout(0.5)(x)

    # 输出层
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.summary()  # 输出模型结构和参数数量
    return model


model = my_model()
#%%
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['AUC'])
# %%
n_epochs = 100
history = model.fit(train_X, train_y,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(test_X, test_y),  # revise
                    validation_freq=1)
#%%
n_epochs = 50
history = model.fit(y_train, y_train_target,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(y_test, y_test_target),  # revise
                    validation_freq=1)  # 测试的epoch间隔数
# %%
predicted_contract = model.predict(y_test)
# %%
yhat = predicted_contract[:, 1]
# %%
y = y_test_target[:, 1]
# %%
from sklearn.metrics import precision_recall_curve
from numpy import argmax

precision, recall, thresholds = precision_recall_curve(y, yhat)
fscore = (2 * precision * recall) / (precision + recall)
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# %%
from sklearn.metrics import classification_report

y_1 = [1 if y > 0.385720 else 0 for y in yhat]
print(classification_report(y_1, y))