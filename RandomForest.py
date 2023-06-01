#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing,Holt
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
#%%
book_data_01 = pd.read_csv('E:/crypto/data/time_group_book20221.csv')
book_data_02 = pd.read_csv('E:/crypto/data/time_group_book20222.csv')
book_data_03 = pd.read_csv('E:/crypto/data/time_group_book20223.csv')
book_data_04 = pd.read_csv('E:/crypto/data/time_group_book20224.csv')
book_data_05 = pd.read_csv('E:/crypto/data/time_group_book20225.csv')
book_data = pd.concat([book_data_01, book_data_02,book_data_03,book_data_04, book_data_05])
book_data['datetime'] = pd.to_datetime(book_data['datetime'])
# book_data = book_data.drop(['Unnamed: 0'], axis=1)
# %%
trade_data_01 = pd.read_csv('E:/crypto/data/time_group_trade20221.csv')
trade_data_02 = pd.read_csv('E:/crypto/data/time_group_trade20222.csv')
trade_data_03 = pd.read_csv('E:/crypto/data/time_group_trade20223.csv')
trade_data_04 = pd.read_csv('E:/crypto/data/time_group_trade20224.csv')
trade_data_05 = pd.read_csv('E:/crypto/data/time_group_trade20225.csv')
trade_data = pd.concat([trade_data_01, trade_data_02, trade_data_03, trade_data_04, trade_data_05])
# trade_data = trade_data.drop(['Unnamed: 0'], axis=1)
trade_data['datetime'] = pd.to_datetime(trade_data['datetime'])
#%%
# def np_move_avg(a,n,mode="valid"):
#     return(np.convolve(a, np.ones((n,))/n, mode=mode))
# trade_data['last_price'] =np_move_avg(np.array(trade_data['last_price']), n=60)
#%%
# trade_data_01 = trade_data.set_index('datetime').resample('1min').apply({'last_price':'last'})
# trade_data_01 = trade_data_01.reset_index()
# trade_data_0101 = trade_data.set_index('datetime').resample('1min').agg(np.mean)
# trade_data_0101 = trade_data_0101.reset_index()
# trade_data_010101 = pd.merge(trade_data_01, trade_data_0101, on='datetime', how='left')
# trade_data_010101 = trade_data_010101.drop(['last_price_y'],axis=1)
#%%
data = pd.merge(trade_data, book_data, on='datetime', how='left')
#%%
# t_time = data.index
# price = data['last_price_vwap']
# plt.figure(figsize=(16,8), dpi=72)
# plt.plot(t_time, price, label='vwap')
# plt.legend(loc=0, frameon=True)
# plt.ylabel('vwap')
# plt.show()
#%%
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
# data = data.fillna(method='bfill')
# data = data.replace(np.inf, 1)
# data = data.replace(-np.inf, -1)

#%%
data = data.set_index('datetime')

data['target'] = np.log(data['last_price_vwap']/data['last_price_vwap'].shift(1))*100
data['target'] = data['target'].shift(-2)
data = data.dropna(axis=0, how='any')
#%%
def check_VIF(df):

    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools import add_constant
    df['c'] = 1
    name = df.columns
    # scaler = MinMaxScaler()
    # df = scaler.fit_transform(df)
    # df = pd.DataFrame(df)
    # X = add_constant(df)
    X = np.matrix(df)
    VIF_list = [variance_inflation_factor(X, i) for i in tqdm(range(X.shape[1]))]
    VIF = pd.DataFrame({'feature':name, 'VIF':VIF_list})
    # VIF = VIF.drop(['c'], axis=0)
    max_VIF = max(VIF)
    print(max_VIF)
    return VIF

VIF_list = check_VIF(data)

name = data.columns
VIF = pd.DataFrame({'feature': name, 'VIF': VIF_list})
#%%
# VIF_list = pd.read_csv('LightGBM_2min_ru_VIF_list_nontarget.csv')
# VIF_list = VIF_list.iloc[:,1:]

col_save = VIF_list.feature[VIF_list.VIF < 0.9]
col_save = col_save.tolist()
#%%
def calcSpearman(data):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:34]):

        ic = data[column].rolling(20).corr(data['target_1'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)

        # print(ic_list)

    return ic_list

IC = calcSpearman(data)

IC = pd.DataFrame(IC)
columns = pd.DataFrame(data.columns)

IC_columns = pd.concat([IC, columns], axis=1)
col = ['value', 'variable']
IC_columns.columns = col

filter_value = 0.01
filter_value2 = -0.01
x_column = IC_columns.variable[IC_columns.value > filter_value]
y_column = IC_columns.variable[IC_columns.value < filter_value2]

x_column = x_column.tolist()
y_column = y_column.tolist()
final_col = x_column+y_column
data = data.reindex(columns=final_col)
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
# train_set = train[:216852]
# test_set = train[192357:]
# train_target = target[:216852]
# test_target = target[192357:]
#%%
train_set = train[train.index <'2022-05-01 00:00:00']
test_set = train[(train.index >= '2022-05-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
train_target = target[train.index <'2022-05-01 00:00:00']
test_target = target[(train.index >= '2022-05-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

X_train = train_set_scaled
X_train_target=train_target
X_test = test_set_scaled
X_test_target =test_target
#%%
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
def LGB_bayesian(
        learning_rate,
        num_leaves,
        bagging_fraction,
        feature_fraction,
        min_child_weight,
        min_child_samples,
        min_split_gain,
        min_data_in_leaf,
        max_depth,
        reg_alpha,
        reg_lambda,
        n_estimators,
        colsample_bytree,
        subsample):
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

model = RandomForestClassifier(n_estimators=100,
                               max_depth=10,
                               min_samples_split=11,
                               min_samples_leaf=12,
                               max_features=0.488105241372471,
                               random_state=2
                               )

import time
start = time.time()

print(X_train.shape)
print(X_test.shape)
model.fit(X_train,X_train_target)
y_pred = model.predict(X_test)

end = time.time()
print('Total Time = %s'%(end-start))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred,X_test_target))
#%%
from sklearn.feature_selection import SelectFromModel
select_model = SelectFromModel(model, prefit=True)
#%%
labels = train.columns
#%%
selected_vars = list(labels[select_model.get_support()])
#%%
import time
start = time.time()

print(X_train.shape)
print(X_test.shape)
model.fit(select_feature,X_train_target)
#%%
y_pred = model.predict(X_test)

end = time.time()
print('Total Time = %s'%(end-start))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred,X_test_target))