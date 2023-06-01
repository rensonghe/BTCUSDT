#%%
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
# month = 1
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
all_data = pd.DataFrame()
symbol = 'ethusdt'
platform = 'gate_swap_u'
year = 2022
for month in tqdm(range(6, 7)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    dataset_base = pq.ParquetDataset('datafile/feat/trade_1s_base/gate_swap_u', filters=filters, filesystem=minio)
    trade_base = dataset_base.read_pandas().to_pandas()
    # trade_base = trade_base.iloc[:, :-3]
    # dataset_vwap = pq.ParquetDataset('datafile/feat/trade_1s_vwap/gate_swap_u', filters=filters, filesystem=minio)
    # trade_vwap = dataset_vwap.read_pandas().to_pandas()
    # trade_vwap = trade_vwap.iloc[:, :-3]
    # dataset_price = pq.ParquetDataset('datafile/feat/depth_1s_price/gate_swap_u', filters=filters, filesystem=minio)
    # depth_price = dataset_price.read_pandas().to_pandas()
    # depth_price = depth_price.iloc[:, :-3]
    # dataset_size = pq.ParquetDataset('datafile/feat/depth_1s_size/gate_swap_u', filters=filters, filesystem=minio)
    # depth_size = dataset_size.read_pandas().to_pandas()
    # depth_size = depth_size.iloc[:, :-3]
    # dataset_wap1 = pq.ParquetDataset('datafile/feat/depth_1s_wap1/gate_swap_u', filters=filters, filesystem=minio)
    # depth_wap1 = dataset_wap1.read_pandas().to_pandas()
    # depth_wap1 = depth_wap1.iloc[:, :-3]
    # dataset_wap2 = pq.ParquetDataset('datafile/feat/depth_1s_wap2/gate_swap_u', filters=filters, filesystem=minio)
    # depth_wap2 = dataset_wap2.read_pandas().to_pandas()
    # depth_wap2 = depth_wap2.iloc[:, :-3]
    # dataset_wap3 = pq.ParquetDataset('datafile/feat/depth_1s_wap3/gate_swap_u', filters=filters, filesystem=minio)
    # depth_wap3 = dataset_wap3.read_pandas().to_pandas()
    # depth_wap3 = depth_wap3.iloc[:, :-3]
    # dataset_wap4 = pq.ParquetDataset('datafile/feat/depth_1s_wap4/gate_swap_u', filters=filters, filesystem=minio)
    # depth_wap4 = dataset_wap4.read_pandas().to_pandas()
    # depth_wap4 = depth_wap4.iloc[:, :-3]
    # dataset_wap5 = pq.ParquetDataset('datafile/feat/depth_1s_wap5/gate_swap_u', filters=filters, filesystem=minio)
    # depth_wap5 = dataset_wap5.read_pandas().to_pandas()
    # depth_wap5 = depth_wap5.iloc[:, :-3]
    # list_of_datasets = [trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5]
    # data_merge = reduce(
    #     lambda left, right: pd.merge(left, right, on=['closetime'], how='inner', suffixes=['', "_drop"]),
    #     list_of_datasets)
    # data_merge.drop([col for col in data_merge.columns if 'drop' in col], axis=1, inplace=True)
    # # all_data = all_data.append(data_merge.sample(frac=0.3, random_state=2022))
    # all_data = all_data.append(data_merge)
    # print(all_data)

#%%
del dataset_wap5, dataset_wap3, dataset_wap4, dataset_wap2, dataset_wap1, dataset_base, dataset_vwap, dataset_price, dataset_size, \
    data_merge, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5, depth_price, list_of_datasets, trade_base, trade_vwap
#%%
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800, unit='s')
#%% 主动买卖因子
# def buy_sell_factor(data):
all_data['buy_side'] = all_data['size'].apply(lambda x : x if x >0 else 0)
#     data['buy_ratio'] = (data.buy_side * data.last_price)/np.sum(data.buy_side * data.last_price)
#
#     return data
test = all_data.set_index('datetime').groupby(pd.Grouper(freq='20s')).apply(buy_sell_factor)

#%%
data = all_data.set_index('datetime').groupby(pd.Grouper(freq='20s')).agg({'last_price':['min','max']})
data.columns = ['_'.join(col) for col in data.columns]
#%%
all_data = all_data.set_index('datetime').groupby(pd.Grouper(freq='20s')).agg('last')
#%%
data = pd.merge(all_data, data, on='datetime', how='left')
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
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator, MFIIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d

MFIIndicator = MFIIndicator(high=data['last_price_max'], low=data['last_price_min'], close=data['last_price'], volume=data['size'], window=20)
data['mfi'] = MFIIndicator.money_flow_index()
forceindex = ForceIndexIndicator(close=data['last_price'], volume=data['size'])
data['forceindex'] = forceindex.force_index()
easyofmove = EaseOfMovementIndicator(high=data['last_price_max'], low=data['last_price_min'], volume=data['size'])
data['easyofmove'] = easyofmove.ease_of_movement()
bollingband = BollingerBands(close=data['last_price'])
data['bollingerhband'] = bollingband.bollinger_hband()
data['bollingerlband'] = bollingband.bollinger_lband()
data['bollingermband'] = bollingband.bollinger_mavg()
data['bollingerpband'] = bollingband.bollinger_pband()
data['bollingerwband'] = bollingband.bollinger_wband()
keltnerchannel = KeltnerChannel(high=data['last_price_max'], low=data['last_price_min'], close=data['last_price'])
data['keltnerhband'] = keltnerchannel.keltner_channel_hband()
data['keltnerlband'] = keltnerchannel.keltner_channel_lband()
data['keltnerwband'] = keltnerchannel.keltner_channel_wband()
data['keltnerpband'] = keltnerchannel.keltner_channel_pband()
donchichannel = DonchianChannel(high=data['last_price_max'], low=data['last_price_min'], close=data['last_price'])
data['donchimband'] = donchichannel.donchian_channel_mband()
data['donchilband'] = donchichannel.donchian_channel_lband()
data['donchipband'] = donchichannel.donchian_channel_pband()
data['donchiwband'] = donchichannel.donchian_channel_wband()
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
#%%
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')
data = data.replace(np.inf, 1)
data = data.replace(-np.inf, -1)
#%%
data = data.set_index('datetime')
# #%%
# data = reduce_mem_usage(data)[0]
# #%%
# target = pd.read_csv('/run/media/ps/data/songhe/crypto/btcusdt_20220101_0731_1min_last_price_vwap.csv')
# target = target.iloc[:,1:]
# target['datetime'] = pd.to_datetime(target['datetime'])
# data = pd.merge(data, target, on='datetime', how='left')
# #%%
# data['target'] = np.log(data['bid_price1']/data['ask_price1'].shift(3))
# data['target'] = data['target'].shift(-3)
# data = data.dropna(axis=0, how='any')
#%%
data['target'] = np.log(data['wap1']/data['wap1'].shift(1))
data['target'] = data['target'].shift(-1)
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
# from colnums import rename_cols_2022_08_14
# cols = list(rename_cols_2022_08_14.values())
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[train.index < '2022-08-01 00:00:00']
# train_set = train[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
test_set = train[(train.index >= '2022-08-01 00:00:00')&(train.index <= '2022-08-31 23:59:59')]
train_target = target[train.index < '2022-08-01 00:00:00']
# train_target = target[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
test_target = target[(train.index >= '2022-08-01 00:00:00')&(train.index <= '2022-08-31 23:59:59')]
# #%%
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
# train_set_scaled = sc.fit_transform(train_set)# 数据归一
# test_set_scaled = sc.transform(test_set)
# train_target = np.array(train_target)
# test_target = np.array(test_target)
#
# X_train = train_set_scaled
# X_train_target=train_target
# X_test = test_set_scaled
# X_test_target =test_target
#%%
X_train = np.array(train_set)
X_train_target = np.array(train_target)
X_test = np.array(test_set)
X_test_target = np.array(test_target)
#%%
del train_set, test_set, train_target, test_target
#%%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
over = SMOTE()
# under = RandomUnderSampler(sampling_strategy=0.5)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
X_train, X_train_target = over.fit_resample(X_train, X_train_target)
#%%
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample):
    # LightGBM expects next three parameters need to be integer.
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    learning_rate = float(learning_rate)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    n_estimators = int(n_estimators)
    min_child_samples = float(min_child_samples)
    min_split_gain = float(min_split_gain)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    kf = TimeSeriesSplit(n_splits=10)
    # kf = GapLeavePOut(p=35000, gap_before=11000, gap_after=24000)
    y_pred = np.zeros(len(X_test_target))
    y_pred_train = np.zeros(len(X_train_target))
    importances = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'min_split_gain': min_split_gain,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'subsample': subsample,
            'n_estimators': n_estimators,
            # 'learning_rate' : learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'binary',
            'save_binary': True,
            'seed': 2022,
            'feature_fraction_seed': 2022,
            'bagging_seed': 2022,
            'drop_seed': 2022,
            'data_random_seed': 2022,
            'boosting_type': 'gbdt',
            'verbose': 1,
            'is_unbalance': False,
            'boost_from_average': True,
            'metric': {'cross_entropy','auc'}}


        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100)
        X_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        # fpr_train, tpr_train, thresholds_train = roc_auc_score(x_val, y_val)
        # gmeans_train = sqrt(tpr_train * (1 - fpr_train))
        # ix_train = argmax(gmeans_train)
        # print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
        #
        # thresholds_point_train = thresholds_train[ix_train]
        # x_val_thresholds = [1 if y > thresholds_point_train else 0 for y in x_val]
        score = roc_auc_score(X_train_target, X_train_pred)

        return score

bounds_LGB = {
    'colsample_bytree': (0.7, 1),
    'n_estimators': (500, 10000),
    'num_leaves': (31, 500),
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.001, 0.3),
    'min_child_weight': (0.00001, 0.01),
    'min_child_samples': (2, 100),
    'min_split_gain': (0.1, 1),
    'subsample': (0.7, 1),
    'reg_alpha': (1, 2),
    'reg_lambda': (1, 2),
    'max_depth': (-1, 50),
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2022)
#%%
init_points = 20
n_iter = 10
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
#%%
LGB_BO.max['target']
#%%
LGB_BO.max['params']
#%%
features = train.columns
features = list(features)
def plot_importance(importances, features_names = features, PLOT_TOP_N = 20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()
#%%
kf = TimeSeriesSplit(n_splits=10)
y_pred = np.zeros(len(X_test_target))
y_pred_train = np.zeros(len(X_train_target))
importances = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = X_train_target[train_index], X_train_target[val_index]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

    # params = {
    #     'boosting_type': 'gbdt',
    #     'metric': {'cross_entropy','auc','average_precision',},
    #     'objective': 'cross_entropy',  # regression,binary,multiclass
    #     # 'num_class': 3,
    #     'seed': 2022,
    #     'feature_fraction_seed': 2022,
    #     'bagging_seed': 2022,
    #     'drop_seed': 2022,
    #     'data_random_seed': 2022,
    #     'num_leaves': int(402.39498468641676),
    #     'learning_rate': float(0.026781560818698175),
    #     'max_depth': int(30.801816172781994),
    #     'n_estimators': int(7360.592472383788),
    #     'bagging_fraction': float(0.8539335878655676),
    #     'feature_fraction': float(0.1477208403984186),
    #     'colsample_bytree': float(0.9161540685016952),
    #     'subsample': float(0.9121060548908404),
    #     'min_child_samples': int(90.49340322410026),
    #     'min_child_weight': float(0.00833244434245706),
    #     'min_split_gain': float(0.6806478505192025),
    #     'min_data_in_leaf': int(174.60092776021733),
    #     'reg_alpha': float(1.289776481431372),
    #     'reg_lambda': float(1.953286864676757),
    #     # 'max_bin': 63,
    #     'save_binary': True,
    #     'verbose': 1,
    #     'is_unbalance': False,
    #     'boost_from_average': True,
    #     # 'cross_entropy':'xentropy'
    # }

    params = {
        'boosting_type': 'gbdt',
        'metric': {'cross_entropy','auc','average_precision',},
        'objective': 'cross_entropy',  # regression,binary,multiclass
        # 'num_class': 3,
        'seed': 2022,
        'feature_fraction_seed': 2022,
        'bagging_seed': 2022,
        'drop_seed': 2022,
        'data_random_seed': 2022,
        'num_leaves': int(LGB_BO.max['params']['num_leaves']),
        'learning_rate': float(LGB_BO.max['params']['learning_rate']),
        'max_depth': int(LGB_BO.max['params']['max_depth']),
        'n_estimators': int(LGB_BO.max['params']['n_estimators']),
        'bagging_fraction': float(LGB_BO.max['params']['bagging_fraction']),
        'feature_fraction': float(LGB_BO.max['params']['feature_fraction']),
        'colsample_bytree': float(LGB_BO.max['params']['colsample_bytree']),
        'subsample': float(LGB_BO.max['params']['subsample']),
        'min_child_samples': int(LGB_BO.max['params']['min_child_samples']),
        'min_child_weight': float(LGB_BO.max['params']['min_child_weight']),
        'min_split_gain': float(LGB_BO.max['params']['min_split_gain']),
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
        'reg_alpha': float(LGB_BO.max['params']['reg_alpha']),
        'reg_lambda': float(LGB_BO.max['params']['reg_lambda']),
        # 'max_bin': 63,
        'save_binary': True,
        'verbose': 1,
        'is_unbalance': False,
        'boost_from_average': True,
        # 'cross_entropy':'xentropy'
    }

    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                      valid_sets=[val_set], verbose_eval=100)

    y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
    importances.append(model.feature_importance(importance_type='gain'))

# plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
# lgb.plot_importance(model, max_num_features=20)
    # a = y_pred[:,1]
# plt.show()
 #%%
y_pred_1 = model.predict(X_test, num_iteration=model.best_iteration)
#%%
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from numpy import sqrt,argmax
fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, y_pred_train)
gmeans_train = sqrt(tpr_train * (1-fpr_train))
ix_train = argmax(gmeans_train)
print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
thresholds_point_train = thresholds_train[ix_train]
yhat_train = [1 if y > thresholds_point_train else 0 for y in y_pred_train]
print("训练集表现：")
print(classification_report(yhat_train,X_train_target))
# print(metrics.confusion_matrix(yhat_train, X_train_target))
#%% roccurve
from sklearn.metrics import roc_curve
from numpy import sqrt,argmax
fpr, tpr, thresholds = roc_curve(X_test_target, y_pred_1)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#%% test set
thresholds_point = thresholds[ix]
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# thresholds_point = thresholds_train[ix_train]
yhat = [1 if y > thresholds_point else 0 for y in y_pred_1]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
# print(metrics.confusion_matrix(yhat, X_test_target))
print('AUC:', metrics.roc_auc_score(yhat, X_test_target))
#%%
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(X_test_target, y_pred)
plt.figure()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision)
plt.show()
#%%
import joblib
joblib.dump(model,'lightGBM_btcusdt_20220809.pkl')
#%%
# joblib.dump(sc, 'scalar_20220531.pkl')
#%%
data = data.reset_index()
#%%
test_data = data[(train.index >= '2022-08-01 00:00:00')&(train.index <= '2022-08-31 23:59:59')]
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred_1,columns=['predict'])
predict['datetime'] = test_data['datetime']
# predict['last_price'] = test_data['last_price']
# predict['bid_price1'] = test_data['bid_price1']
# predict['ask_price1'] = test_data['ask_price1']
predict['wap1'] = test_data['wap1']
predict['target'] = test_data['target']
#%%
print(predict['predict'].describe())
#%%
df_1 = predict.loc[predict['predict']>0.6]
df_0 = predict.loc[predict['predict']<0.4]
print(len(df_1))
print(len(df_0))
#%%
test_1 = len(df_1[(df_1.predict>0.6)&(df_1.target>0)])/len(df_1)
df_0['target'] = np.where(df_0['target']>0,1,-1)
test_0 = len(df_0[(df_0.predict<0.4)&(df_0.target<0)])/len(df_0)
print(test_1)
print(test_0)
#%%
model.save_model('btcusdt_20220101_0322_1min_lightGBM.pkl')
#%%
predict.to_csv('ethusdt_20220801_0831_20s_wap1_strategy_8.0_20220909.csv')
#%% 15min采样
predict['datetime'] = pd.to_datetime(predict['datetime'])
# predict_15min = predict[predict.datetime>='2022-3-7']
predict_15min = predict.set_index('datetime').resample('15min').apply('last')
predict_15min['Predict'] = np.where(predict_15min['Predict']>0,1,-1)
predict_15min['target'] = np.where(predict_15min['target']>0,1,-1)
test = len(predict_15min[(predict_15min.Predict>0)&(predict_15min.target>0)|((predict_15min.Predict<0)&(predict_15min.target<0))])/len(predict_15min)
print(test)
#%%
predict_15min.to_csv('btcusdt_20220615_0630_15min_vwap_strategy_1.0_20220726.csv')
#%%
from scipy.stats import ks_2samp
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
#%%
D_value = 1.36*np.sqrt((len(X_train)+len(X_test))/(len(X_train)*len(X_test)))
print(D_value)
#%%
for i in range(39):
    print(i)
    print(adfuller(X_test[:,i]))
#%%
for i in range(80):
    print(i)
    print(ks_2samp(X_train[:,i],X_test[:,i]))
#%%
D_value = 1.36*np.sqrt((len(X_train_target)+len(X_test_target))/(len(X_train_target)*len(X_test_target)))
print(D_value)
#%%
ks_2samp(X_train_target,X_test_target)
#%%
import matplotlib.pyplot as plt

data.loc[:100,['bid_price1']].plot(labels='bid_price1')
data.loc[:100,['ask_price1']].plot(labels='ask_price1')
data.loc[:100,['wap1']].plot(labels='wap1')
plt.legend()
plt.show()