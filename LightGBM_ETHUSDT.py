#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
from tqdm import tqdm
import seaborn as sns
import warnings
import keras
import joblib
from scipy.stats import ks_2samp
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
all_data = pd.DataFrame()
symbol = 'ethusdt'
platform = 'gate_swap_u'
year = 2022
for month in tqdm(range(6, 12)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    order_book_base = pq.ParquetDataset('datafile/tick/order_book_100ms/gate_swap_u', filters=filters, filesystem=minio)
    orderbook = order_book_base.read_pandas().to_pandas()
    orderbook = orderbook.reset_index(drop=False)
    orderbook = orderbook.iloc[:, :-6]
    trade_base = pq.ParquetDataset('datafile/tick/trade/gate_swap_u', filters=filters, filesystem=minio)
    trade = trade_base.read_pandas().to_pandas()
    trade = trade.sort_values(by='dealid', ascending=True)
    trade = trade.iloc[:, 1:-3]
    # list_of_datasets = [order_book_base, trade]
    # data_merge = reduce(
    #     lambda left, right: pd.merge(left, right, on=['closetime'], how='inner', suffixes=['', "_drop"]),
    #     list_of_datasets)
    # data_merge.drop([col for col in data_merge.columns if 'drop' in col], axis=1, inplace=True)
    # # all_data = all_data.append(data_merge.sample(frac=0.3, random_state=2022))
    # all_data = all_data.append(data_merge)
    # print(all_data)

#%%
del order_book_base, orderbook, trade_base, trade
#%%
# all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800000, unit='ms')
#%%
add_factor['vwap'] = (add_factor['price'].fillna(0)*abs(add_factor['size'].fillna(0))).rolling(120).sum()/abs(add_factor['size'].fillna(0)).rolling(120).sum()
# add_factor['vwap'] = (add_factor['wap1'].fillna(0)*abs(add_factor['size'].fillna(0))).rolling(120).sum()/abs(add_factor['size'].fillna(0)).rolling(120).sum()
#%% time bar
add_factor['datetime'] = pd.to_datetime(add_factor['closetime']+28800000, unit='ms')
data = add_factor.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply('last')
data = data.dropna(axis=0)
#%% volume/dollar bar
def volume_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        dv_column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def volume_bar_df(df, dv_column, m):
    idx = volume_bars(df, dv_column, m)
    return df.iloc[idx].drop_duplicates()

tick_data = add_factor.copy()
tick_data['cum_size'] = tick_data['cum_size'].fillna(0)
data = volume_bar_df(tick_data, 'cum_size',50_000_000)
#%%
price_max_min = add_factor.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg({'price':['min','max'], 'closetime':'last'})
price_max_min.columns = ['_'.join(col) for col in price_max_min.columns]
price_max_min = price_max_min.rename({'closetime_last':'closetime', 'price_max':'high', 'price_min':'low'},axis='columns')
price_max_min = price_max_min.dropna(axis=0)
#%%
data = pd.merge(data, price_max_min, on='closetime', how='left')
data['datetime'] = pd.to_datetime(data['closetime']+28800000, unit='ms')
data = data.set_index('datetime')
#%%
from ta.volume import *
from ta.volatility import *
from ta.trend import *
from ta.momentum import *
def indicator_function(df):
    df['indicator_FI'] = force_index(close=df['price'], volume=df['size'], window=13)
    df['indicator_EoM'] = ease_of_movement(high=df['high'], low=df['low'], volume=df['size'], window=14)
    df['indicator_VPT'] = volume_price_trend(close=df['price'], volume=df['size'])
    df['indicator_BBH'] = bollinger_hband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBL'] = bollinger_lband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBM'] = bollinger_mavg(close=df['price'], window=20)
    df['indicator_BBP'] = bollinger_pband(close=df['price'], window=20, window_dev=2)
    df['indicator_BBW'] = bollinger_wband(close=df['price'], window=20, window_dev=2)
    df['indicator_DCH'] = donchian_channel_hband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCL'] = donchian_channel_lband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCM'] = donchian_channel_mband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCW'] = donchian_channel_wband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_DCP'] = donchian_channel_pband(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['indicator_KCL'] = keltner_channel_lband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCH'] = keltner_channel_hband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCW'] = keltner_channel_wband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCP'] = keltner_channel_pband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_KCM'] = keltner_channel_mband(high=df['high'], low=df['low'], close=df['price'], window=10)
    df['indicator_MACD'] = macd(close=df['price'], window_fast=26, window_slow=12)
    df['indicator_MACD_diff'] = macd_diff(close=df['price'], window_fast=26, window_slow=12, window_sign=9)
    df['indicator_MACD_signal'] = macd_signal(close=df['price'], window_fast=26, window_slow=12, window_sign=9)
    df['indicator_SMA_fast'] = sma_indicator(close=df['price'], window=16)
    df['indicator_SMA_slow'] = sma_indicator(close=df['price'], window=32)
    df['indicator_RSI'] = stochrsi(close=df['price'], window=9, smooth1=16, smooth2=26)
    df['indicator_RSIK'] = stochrsi_k(close=df['price'], window=9, smooth1=16, smooth2=26)
    df['indicator_RSID'] = stochrsi_d(close=df['price'], window=9, smooth1=16, smooth2=26)

    return df
#%%
# data = indicator_function(data)
#%%
data['mid'] = (data['ask_price1']+data['bid_price1'])/2
#%%
df = data.iloc[:10,:]
# df['ask'] = df['ask_price1']*(1-0.00005)
# df['bid'] = df['bid_price1']*(1+0.00005)
df[['ask_price1', 'bid_price1','wap1','price']].plot()
plt.show()
#%%
# data['mid'] = (data['ask_price1']+data['bid_price1'])/2
data['buy_success'] = (data['bid_price1'].shift(1))*1.00003-data['price_min']
data['sell_success'] = data['price_max']-(data['ask_price1'].shift(1))*0.99997
data['buy_success'] = np.where(data['buy_success']>0, 1, data['buy_success'])
data['sell_success'] = np.where(data['sell_success']>0, 1, data['sell_success'])

data['target'] = data['buy_success']*data['sell_success']
print(len(data[data['target'].isnull().values==True])/len(data['target']))
print(len(data[data['target']==1])/len(data['target']))
#%%
def getDailyVol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    a = df0 -1 #using a variable to avoid the error message.
    df0=pd.Series(close.index[a],
                  index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def get_Daily_Volatility(close,span0=20):
    # simple percentage returns
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0

#set the boundary of barriers, based on 15min EWM
daily_volatility = get_Daily_Volatility(data['price'], span0=15)
# how many times we hold the stock which set the vertical barrier
t_final = 15
#the up and low boundary multipliers
upper_lower_multipliers = [3, 1]
#allign the index
prices = data.loc[daily_volatility.index]
#%%
def get_3_barriers():
    #create a container
    barriers = pd.DataFrame(columns=['days_passed',
              'price', 'vert_barrier',
              'top_barrier', 'bottom_barrier'],
               index = daily_volatility.index)
    for day, vol in (tqdm(daily_volatility.iteritems())):
        days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
        #set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index)and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan
        #set the top barrier
        close = prices['price']
        if upper_lower_multipliers[0] > 0:
            top_barrier = close[day] + close[day] * upper_lower_multipliers[0] * vol
        else:
            #set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        #set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = close[day] - close[day] * upper_lower_multipliers[1] * vol
        else:
            #set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
        barriers.loc[day, ['days_passed', 'price','vert_barrier','top_barrier', 'bottom_barrier']] = days_passed, prices['price'], vert_barrier,top_barrier, bottom_barrier
    return barriers
barriers = get_3_barriers()
#%%
# data['target'] = data['bid_price1'].shift(1)-data['ask_price1']
data['target'] = np.log(data['vwap']/data['vwap'].shift(5))
# data = data.drop(['buy_success', 'sell_success'], axis=1)
data['target'] = data['target'].shift(-5)
# data = data.dropna(axis=1, how='any')
#%%
def classify(y):

    if y < -0.001:
        return 0
    if y > 0.001:
        return 1
    else:
        return -1
#%%
data['target'] = data['target'].apply(lambda x:classify(x))
# data['target'] = abs(data['target']).apply(lambda x:classify(x))
print(data['target'].value_counts())
print(len(data[data['target']==-1])/len(data['target']))
#%%
data = data[~data['target'].isin([-1])]
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)
#%%
train = data[train_col]
train = train.iloc[:,25:66]
target = data["target"] # 取前26列为训练数据，最后一列为target
#%%
train = data[data.index < '2022-08-01 00:00:00']
test = data[(data.index >= '2022-08-01 00:00:00')&(data.index <= '2022-08-30 23:59:59')]
train['target'] = train['target'].apply(lambda x:classify(x))
train = train[~train['target'].isin([-1])]
train_set = train[train_col]
train_set = train_set.iloc[:,25:66]
# train_set = train_set.iloc[:,45:86] #binance
train_target = train["target"]
test_set = test[train_col]
test_set = test_set.iloc[:,25:66]
# test_set = test_set.iloc[:,45:86]
test_target = test["target"]
#%%
train_set = train[train.index < '2022-11-01 00:00:00']
# train_set = train[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
test_set = train[(train.index >= '2022-11-01 00:00:00')&(train.index <= '2022-11-30 23:59:59')]
train_target = target[train.index < '2022-11-01 00:00:00']
# train_target = target[(train.index >= '2022-05-15 00:00:00')&(train.index <= '2022-06-15 23:59:59')]
test_target = target[(train.index >= '2022-11-01 00:00:00')&(train.index <= '2022-11-30 23:59:59')]
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
from keras.utils.np_utils import to_categorical
X_train_target = to_categorical(X_train_target, num_classes=3)
X_test_target = to_categorical(X_test_target, num_classes=3)
#%%
del train_set, test_set, train_target, test_target
#%%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
over = SMOTE(random_state=2023)
# under = RandomUnderSampler(sampling_strategy=0.5)
X_train, X_train_target = over.fit_resample(X_train, X_train_target)
#%%
def custom_smooth_l1_loss_eval(y_pred, lgb_train):
    """
    Calculate loss value of the custom loss function
     Args:
        y_true : array-like of shape = [n_samples] The target values.
        y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        loss: loss value
        is_higher_better : bool, loss是越低越好，所以这个返回为False
        Is eval result higher better, e.g. AUC is ``is_higher_better``.
    """
    y_true = lgb_train.get_label()
    # y_pred = y_pred.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    loss = np.where(np.abs(residual) < 1, (residual ** 2) * 0.5, np.abs(residual) - 0.5)
    return "custom_asymmetric_eval", np.mean(loss), False

def custom_smooth_l1_loss_train(y_pred, lgb_train):
    """Calculate smooth_l1_loss
    Args:
        y_true : array-like of shape = [n_samples]
        The target values. y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        grad: gradient, should be list, numpy 1-D array or pandas Series
        hess: matrix hessian value
    """
    y_true = lgb_train.get_label()
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    residual = (y_true - y_pred).astype("float")
    grad = np.where(np.abs(residual) < 1, residual, 1)
    hess = np.where(np.abs(residual) < 1, 1.0, 0.0)
    return grad, hess
#%%
from sklearn.utils.class_weight import compute_class_weight
# class_w = {1 : 3, 0 : 1}
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples,
        min_split_gain, min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample,
                 ):
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
    # scale_pos_weight = float(scale_pos_weight)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    kf = TimeSeriesSplit(n_splits=5)
    X_train_pred = np.zeros(len(X_train_target))


    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = X_train_target[train_index], X_train_target[val_index]
        # sample_x = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # sample_x = [1 if i == 0 else 3 for i in y_train.tolist()]
        # sample_y = compute_class_weight(class_weight='balanced', classes=np.unique(y_val), y=y_val)
        # sample_y = [1 if i == 0 else 3 for i in y_val.tolist()]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val)
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
            'objective': 'cross_entropy',
            # 'objective': 'multiclass',
            # 'num_class': '3',
            'save_binary': True,
            'seed': 2023,
            'feature_fraction_seed': 2023,
            'bagging_seed': 2023,
            'drop_seed': 2023,
            'data_random_seed': 2023,
            'boosting_type': 'gbdt',
            'verbose': 1,
            # 'is_unbalance': True,
            # 'scale_pos_weight': 20,
            'boost_from_average': True,
            'metric': {'cross_entropy','auc'},
            # 'metric': {'multi_logloss','auc'},
            'num_threads': 28}


        model = lgb.train(params, train_set=train_set, num_boost_round=5000, early_stopping_rounds=50,
                          valid_sets=[val_set], verbose_eval=100) #fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)
        X_train_pred += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
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
    # 'scale_pos_weight':(0.5, 10)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=2023)

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
kf = TimeSeriesSplit(n_splits=5)
y_pred = np.zeros(len(X_test_target))
y_pred_train = np.zeros(len(X_train_target))
importances = []
model_list = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
    print('Model:',fold)
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = X_train_target[train_index], X_train_target[val_index]
    # train_weight = [1 if i == 0 else 3 for i in y_train.tolist()]
    # test_weight = [1 if i == 0 else 3 for i in y_val.tolist()]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)


    params = {
        'boosting_type': 'gbdt',
        # 'metric': 'multi_logloss',
        # 'objective': 'multiclass',
        'metric': {'cross_entropy','auc','average_precision',},
        'objective': 'binary',  # regression,binary,multiclass
        # 'num_class': 3,
        'seed': 2023,
        'feature_fraction_seed': 2023,
        'bagging_seed': 2023,
        'drop_seed': 2023,
        'data_random_seed': 2023,
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
        # 'is_unbalance': True,
        # 'scale_pos_weight': 20,
        'boost_from_average': True,
        # 'cross_entropy':'xentropy'
        'num_threads': 28
    }

    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                      valid_sets=[val_set], verbose_eval=100)#fobj=custom_smooth_l1_loss_train, feval=custom_smooth_l1_loss_eval)

    y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
    importances.append(model.feature_importance(importance_type='gain'))
    model_list.append(model)

# plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
# lgb.plot_importance(model, max_num_features=20)
    # a = y_pred[:,1]
# plt.show()
#%%
# y_pred_1 = model.predict(X_test, num_iteration=model.best_iteration)
# y_pred_train_1 = model.predict(X_train, num_iteration=model.best_iteration)
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
from sklearn.metrics import roc_curve,precision_recall_curve
from numpy import sqrt,argmax
fpr, tpr, thresholds = roc_curve(X_test_target, y_pred)
# pr, tpr, thresholds = precision_recall_curve(X_test_target, y_pred)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
thresholds_point = thresholds[ix]
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# thresholds_point = thresholds_train[ix_train]
yhat = [1 if y > thresholds_point else 0 for y in y_pred]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
# print(metrics.confusion_matrix(yhat, X_test_target))
print('AUC:', metrics.roc_auc_score(yhat, X_test_target))
#%%
import joblib
joblib.dump(model_list[0],'15min_vwap_ethusdt_lightGBM_20230103_0.pkl')
joblib.dump(model_list[1],'15min_vwap_ethusdt_lightGBM_20230103_1.pkl')
joblib.dump(model_list[2],'15min_vwap_ethusdt_lightGBM_20230103_2.pkl')
joblib.dump(model_list[3],'15min_vwap_ethusdt_lightGBM_20230103_3.pkl')
joblib.dump(model_list[4],'15min_vwap_ethusdt_lightGBM_20230103_4.pkl')
#%%
data = data.reset_index()
#%%
test_data = test
# test_data = test2p
# test_data = data[(train.index >= '2022-11-01 00:00:00')&(train.index <= '2022-11-30 23:59:59')]
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred,columns=['predict'])
# predict = pd.DataFrame(y_pred,columns=['p0','p1','p2'])
predict['closetime'] = test_data['closetime']
# predict['mid_price'] = test_data['mid_price']
predict['vwap'] = test_data['vwap']
predict['close'] = test_data['price']
predict['symbol'] = 'ethusdt'
predict['platform'] = 'gate_swap_u'
predict['starttime'] = test_data['closetime']
predict['endtime'] = predict['starttime']+60000*60*2
# predict['side'] = 'buy_sell'
predict['target'] = test_data['target']
#%%
predict['pctrank'] = predict.index.map(lambda x : predict.loc[:x].predict.rank(pct=True)[x])
#%%
df_1 = predict.loc[predict['predict']>np.percentile(y_pred_train, 95)]
df_0 = predict.loc[predict['predict']<np.percentile(y_pred_train, 5)]
print(len(df_1))
print(len(df_0))
#%%
df_1 = predict.loc[predict['pctrank']>0.98]
df_0 = predict.loc[predict['pctrank']<0.02]
print(len(df_1))
print(len(df_0))

#%%
predict.to_csv('/songhe/ETHUSDT/ethusdt_20221201_1230_15min_vwap_ST2.0_20230115_first.csv')
#%%
delete_list = []
for i in range(len(X_train)):
    pvalue = adfuller(X_train[i], autolag='AIC')[1]
    t_test = adfuller(X_train[i], autolag='AIC')[0]
    interval_10 = adfuller(X_train[i], autolag='AIC')[4]['10%']
    # print(adfuller(X_train[i], autolag='AIC'))
    if pvalue > 0.05 or t_test > interval_10:
        delete_list.append(i)
    # print(i)
    # print(adfuller(X_train[:,i]))

train = train.drop(train.columns[delete_list], axis=1)
#%%
df_1['side'] = 'buy'
df_0['side'] = 'sell'
final_df = pd.concat([df_1, df_0], axis=0)
final_df = final_df.sort_values(by='closetime', ascending=True)
final_df = final_df.reset_index(drop=True)
print(final_df.loc[:,['target','predict']].corr())
# print(stats.jarque_bera(final_df['predict']))
#%%
from pyarrow import Table
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
dir_name = 'datafile/bt_record/songhe/{}'.format("btid=hft100ms_gate_15min_vwap_20221201_1230_20230118_xrpusdt")
pq.write_to_dataset(Table.from_pandas(df=final_df),
                    root_path=dir_name,
                    filesystem=minio, basename_template="part-{i}.parquet",
                    existing_data_behavior="overwrite_or_ignore")
#%%
# final_df = final_df.dropna(axis=0)
final_df['datetime'] = pd.to_datetime(final_df['closetime']+28800000, unit='ms')
final_df = final_df.set_index('datetime')
# final_df = final_df[final_df.index>='2022-11-12 21:00:00']
final_df.to_csv('/songhe/ETHUSDT/ethusdt_20220801_0830_5bar_vwap_ST2.0_20230130_filter.csv')
#%%
test2 = test.reset_index()
test2['target'] = predict['predict']
#%%
test2p = test2[(test2['target']<=np.percentile(y_pred_train, 5))|(test2['target']>=np.percentile(y_pred_train, 95))]
test2p.loc[test2p['target']>=0.5,'target'] = 1
test2p.loc[test2p['target']<=0.5,'target'] = 0
test2p = test2p.set_index('datetime')
#%%
train2p = pd.concat([train, test2p], axis=0)
train_set = train2p.iloc[:,25:66]
# train_set = train_set.iloc[:,45:86] #binance
train_target = train2p["target"]
test_set = test2p.iloc[:,25:66]
# test_set = test_set.iloc[:,45:86]
test_target = test2p["target"]

