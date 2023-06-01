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
    # dataset_vwap = pq.ParquetDataset('/songhe/AI1.0/data/feat/trade_1s_vwap/gate_swap_u', filters=filters)
    # trade_vwap = dataset_vwap.read_pandas().to_pandas()
    # trade_vwap = trade_vwap.iloc[:, :-3]
    # dataset_price = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_price/gate_swap_u',filters=filters)
    # depth_price = dataset_price.read_pandas().to_pandas()
    # depth_price = depth_price.iloc[:, :-3]
    # dataset_size = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_size/gate_swap_u', filters=filters)
    # depth_size = dataset_size.read_pandas().to_pandas()
    # depth_size = depth_size.iloc[:, :-3]
    # dataset_wap1 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap1/gate_swap_u', filters=filters)
    # depth_wap1 = dataset_wap1.read_pandas().to_pandas()
    # depth_wap1 = depth_wap1.iloc[:, :-3]
    # dataset_wap2 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap2/gate_swap_u', filters=filters)
    # depth_wap2 = dataset_wap2.read_pandas().to_pandas()
    # depth_wap2 = depth_wap2.iloc[:, :-3]
    # dataset_wap3 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap3/gate_swap_u', filters=filters)
    # depth_wap3 = dataset_wap3.read_pandas().to_pandas()
    # depth_wap3 = depth_wap3.iloc[:, :-3]
    # dataset_wap4 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap4/gate_swap_u', filters=filters)
    # depth_wap4 = dataset_wap4.read_pandas().to_pandas()
    # depth_wap4 = depth_wap4.iloc[:, :-3]
    # dataset_wap5 = pq.ParquetDataset('/songhe/AI1.0/data/feat/depth_1s_wap5/gate_swap_u', filters=filters)
    # depth_wap5 = dataset_wap5.read_pandas().to_pandas()
    # depth_wap5 = depth_wap5.iloc[:, :-3]
    # list_of_datasets = [trade_base, trade_vwap, depth_price, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5]
    # data_merge = reduce(
    #     lambda left, right: pd.merge(left, right, on=['closetime'], how='inner', suffixes=['', "_drop"]),
    #     list_of_datasets)
    # data_merge.drop([col for col in data_merge.columns if 'drop' in col], axis=1, inplace=True)
    # all_data = all_data.append(data_merge.sample(frac=0.4, random_state=2022))
    # all_data = all_data.append(data_merge)
    all_data = all_data.append(trade_base)

#%%
del dataset_wap5, dataset_wap3, dataset_wap4, dataset_wap2, dataset_wap1, dataset_base, dataset_vwap, dataset_price, dataset_size, \
    data_merge, depth_size, depth_wap1, depth_wap2, depth_wap3, depth_wap4, depth_wap5, depth_price, list_of_datasets, trade_base, trade_vwap
#%%
all_data['datetime'] = pd.to_datetime(all_data['closetime']+28800, unit='s')
#%%
all_data = all_data[(all_data.datetime>='2022-01-01 08:00:01')]
#%%
all_data = all_data.set_index('datetime')
#%%
last_price = all_data['last_price']
target = np.log(last_price/last_price.shift(60))
target = target.shift(-60)
# target = target.dropna()
#%%
def classify(y):

    if y < 0:
        return 0
    # if y > 0:
    #     return 1
    else:
        return 1
target = target.apply(lambda x:classify(x))
print(target.value_counts())
#%%
from colnums import rename_cols_2022_08_14
cols = list(rename_cols_2022_08_14.values())

# cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = all_data[train_col]
#%%
del all_data, last_price
#%%
train = train.fillna(method='ffill')
train = train.fillna(method='bfill')
train = train.replace(np.inf, 1)
train = train.replace(-np.inf, -1)
#%%
train['target'] = target
#%%
train = train[~train['target'].isin([-1])]
#%%
target = train['target']
train = train[train_col]
#%%
train_set = train[train.index < '2022-08-18 08:00:00']
train_target = target[train.index < '2022-08-18 08:00:00']
X_train = np.array(train_set)
X_train_target = np.array(train_target)
#%%
def LGB_bayesian(learning_rate, num_leaves, bagging_fraction, feature_fraction, min_child_weight, min_child_samples, min_split_gain,
        min_data_in_leaf, max_depth, reg_alpha, reg_lambda, n_estimators, colsample_bytree, subsample):
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
    # y_pred_train = np.zeros(len(X_train_target))
    # importances = []

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
#%%
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
n_iter = 20
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
#%%
LGB_BO.max['target']
#%%
LGB_BO.max['params']
#%%
kf = TimeSeriesSplit(n_splits=10)
y_pred_train = np.zeros(len(X_train_target))
importances = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = X_train_target[train_index], X_train_target[val_index]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

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

    # y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
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
#%%
import joblib

joblib.dump(model, 'lightGBM_test_20220818.pkl')