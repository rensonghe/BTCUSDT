import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from datetime import datetime,timedelta
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# #%%
# import sys
# sys.path.append('/home/admin/notebooks/crossborder/')
# # from getlabel import getdata_order_v1, getdata_measurable_indicator_v1,getdata_model,getdata_order_v2
# from feature_process_lgb import pdf_preprocess,getdata_feature
# # import getdata_indicator
#%%
# t_time = data.index
# price = data['last_price_vwap']
# plt.figure(figsize=(16,8), dpi=72)
# plt.plot(t_time, price, label='vwap')
# plt.legend(loc=0, frameon=True)
# plt.ylabel('vwap')
# plt.show()

#%%
data = data.set_index('datetime')
#%%
data['target'] = np.log(data['last_price_vwap']/data['last_price_vwap'].shift(1))*100
data['target'] = data['target'].shift(-2)
#%%
sparkSession = SparkSession \
        .builder \
        .appName("bigbear Spark") \
        .enableHiveSupport() \
        .getOrCreate()
#%%
def fillNan(df, num_vars, cat_vars):
    """
    将DataFrame数据空值填充，并将类别型特征转换成数值型
    """
    df[num_vars] = df[num_vars].fillna(-99999999)  # 数值型填充
    df[cat_vars] = df[cat_vars].fillna('Fillothers')  # 类别型填充
    return df


def generate_binomial_mask(B, T, p=0.25):
    return np.random.binomial(1, p, size=(B, T))  # 二项分布中抽取样本


def featureImportsChooseFeatures(df, df_test, features, k=100, p=0.25, iscatboost=False, cat_features=None,
                                 featureNum=0):
    catFeatureDict = dict(zip(features, range(len(features))))

    baseModel = CatBoostClassifier(iterations=20,
                                   learning_rate=0.1,
                                   depth=5,
                                   cat_features=[catFeatureDict[i] for i in cat_features],
                                   loss_function='Logloss',
                                   silent=True)

    X_train = df[features]
    X_test = df_test[features]
    y_train = df['target']
    y_test = df_test['target']

    mask = generate_binomial_mask(k, len(features), p=p)
    mask = np.array(mask, dtype=bool)
    featureInImports = {}
    featureOutImports = {}
    featureGiniImports = {}
    for i in features:
        featureInImports[i] = [0, 0]
        featureOutImports[i] = [0, 0]
        featureGiniImports[i] = [0, 0]

    for _, (i, feature) in zip(range(len(mask)), enumerate(mask)):
        feature = np.array(features)[feature]  # 选中的特征
        df_tmp = X_train[feature]
        test_tmp = X_test[feature]
        tmpCatFeatures = dict(zip(feature, range(len(feature))))
        model = CatBoostClassifier(iterations=20,
                                   learning_rate=0.1,
                                   depth=5,
                                   cat_features=[tmpCatFeatures[i] for i in cat_features if i in feature],
                                   loss_function='Logloss',
                                   silent=True)
        model.fit(df_tmp, y_train)  # 选择特征建模
        oobScore = roc_auc_score(y_test, model.predict_proba(test_tmp)[:, 1])
        OutFeature = set(features) - set(feature)
        GiniImports = model.feature_importances_
        for num, i in enumerate(feature):
            featureInImports[i] = [featureInImports[i][0] + oobScore, featureInImports[i][1] + 1]  # 选中的特征OOB得分(模型auc)
            featureGiniImports[i] = [featureGiniImports[i][0] + GiniImports[num],
                                     featureGiniImports[i][1] + 1]  # 特征的gini得分
        for i in OutFeature:
            featureOutImports[i] = [featureOutImports[i][0] + oobScore, featureOutImports[i][1] + 1]  # 没有选中特征得分

    featureDropImports = {}
    featureGiniAvgImports = {}
    for feature in features:
        featureDropImports[feature] = featureInImports[feature][0] / featureInImports[feature][1] - \
                                      featureOutImports[feature][0] / featureOutImports[feature][
                                          1]  # 选择这个特征得分 - 没有选中这个特征得分
        featureGiniAvgImports[feature] = featureGiniImports[feature][0] / (featureGiniImports[feature][1])  # gini平均分

    dt = pd.DataFrame()
    dt['features'] = features
    dt['featureDropImports'] = dt['features'].map(featureDropImports)
    dt['featureGiniAvgImports'] = dt['features'].map(featureGiniAvgImports)
    dt['featuresAllSocre'] = dt['features'].map(featureDropImports) + dt['features'].map(featureGiniAvgImports)

    featureDropImports = sorted(featureDropImports.items(), key=lambda x: x[1], reverse=True)
    featureGiniAvgImports = sorted(featureGiniAvgImports.items(), key=lambda x: x[1], reverse=True)

    temp_ = pd.DataFrame(dt['featureDropImports'])
    temp_ = temp_.sort_values(by='featureDropImports', ascending=False).reindex()
    threhold_index = int(len(temp_) * 0.9)
    threhold = temp_['featureDropImports'].iloc[threhold_index]

    ginithrehold = dt.featureGiniAvgImports.mean() / 30

    if threhold_index > featureNum:
        if threhold < 0:
            feature_num_ = list(
                dt[(dt['featureDropImports'] > threhold) & (dt['featureGiniAvgImports'] > ginithrehold)]['features'])
        else:
            feature_num_ = list(
                dt[(dt['featureDropImports'] >= 0) & (dt['featureGiniAvgImports'] > ginithrehold)]['features'])
        if len(feature_num_) < featureNum:
            feature_num = list(dt.sort_values(by='featuresAllSocre', ascending=False)['features'][:featureNum])
        else:
            feature_num = feature_num_
    else:
        feature_num = list(dt.sort_values(by='featuresAllSocre', ascending=False)['features'][:featureNum])

    cat_features = list(set(cat_features).intersection(set(feature_num)))
    tmp_CatFeatures = dict(zip(feature_num, range(len(feature_num))))
    #         if iscatboost:
    Model_choose = CatBoostClassifier(iterations=20,
                                      learning_rate=0.1,
                                      depth=5,
                                      cat_features=[tmp_CatFeatures[i] for i in cat_features if i in feature_num],
                                      loss_function='Logloss',
                                      silent=True)
    #         else:
    #             Model_choose = clone(baseModel)

    Model_choose.fit(df[feature_num], df['target'])  # 特征选择后的模型
    baseModel.fit(df[features], df['target'])  # 原始全特征模型

    testAllAUC = roc_auc_score(df_test['target'], baseModel.predict_proba(df_test[features])[:, 1])  # 全特征效果
    testChooseAUC = roc_auc_score(df_test['target'], Model_choose.predict_proba(df_test[feature_num])[:, 1])  # 选择特征效果

    from sklearn.metrics import classification_report, roc_curve
    from numpy import sqrt
    from numpy import argmax
    fpr, tpr, thresholds = roc_curve(df_test['target'], Model_choose.predict_proba(df_test[feature_num])[:, 1])
    gmeans = sqrt(tpr * (1 - fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    yhat = [1 if y > thresholds[ix] else 0 for y in Model_choose.predict_proba(df_test[feature_num])[:, 1]]
    classification_report = classification_report(yhat, df_test['target'])
    print(
        f'TestAUC: Origin features num:{len(features)}, AUC:{testAllAUC}; Choose features num:{len(feature_num)}, AUC:{testChooseAUC};classification_report:{classification_report}')


    if len(feature_num) > featureNum:
        featureDropImports, featureGiniAvgImports, feature_num = featureImportsChooseFeatures(
            df[feature_num + ['target']]
            , df_test[feature_num + ['target']]
            , features=feature_num
            , k=k
            , p=p
            , iscatboost=iscatboost
            , cat_features=cat_features
            , featureNum=featureNum
        )
    return featureDropImports, featureGiniAvgImports, feature_num



# %%
# df_train, df_test = getdata_model(sparkSession, 'v01003')
# df_train = df_train[df_train['identityriskscore'].notnull()]
# df_test = df_test[df_test['identityriskscore'].notnull()]

# df_train = df_train[df_train['label'] != "-1"]
# df_test = df_test[df_test['label'] != "-1"]

# df_train, df_test = pdf_preprocess(df_train, df_test, 'v01003')
#
# init_cols = df_train.columns
# df_train_trans, df_test_trans, need_columns_final, num_vars, cat_vars, dict_cat_vars_map = getdata_feature(df_train,
#                                                                                                            df_test,
#                                                                                                            init_cols)
#
# df_train['label'] = df_train.label
# df_test['label'] = df_test.label
# # 去除-1类别，保持二分类
#
# label = 'label'
# features = list(df1.columns)
# features.remove(label)

# 训练集
# autoFeat = AutoFeatures(df_train_trans, features, label)

# 测试训练集，空值和类别型转换
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
train_set = train[train.index <'2022-02-01 00:00:00']
# train_set = train[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
test_set = train[(train.index >= '2022-02-01 00:00:00')&(train.index <= '2022-02-14 23:59:59')]
train_target = target[train.index <'2022-02-01 00:00:00']
# train_target = target[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
test_target = target[(train.index >= '2022-02-01 00:00:00')&(train.index <= '2022-02-14 23:59:59')]
#%%
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
# %%
df_train_trans = data[data.index <'2022-02-15 00:00:00']
df_test_trans = data[(data.index >= '2022-02-15 00:00:00')&(data.index <= '2022-02-28 23:59:59')]
#%%
target = 'target'
features = list(data.columns)
features.remove(target)
# %%
# # # 特征筛选
featuresImport = featureImportsChooseFeatures(df=df_train_trans
                                              , df_test=df_test_trans
                                              , features=features
                                              , k=80
                                              , iscatboost=True
                                              , cat_features=[]
                                              , featureNum=40
                                              )
# %%
final_features = featuresImport[2]
#%%
data = data.set_index('datetime')

data['target'] = np.log(data['last_price_vwap']/data['last_price_vwap'].shift(1))*100
data['target'] = data['target'].shift(-2)
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
train = train.reindex(columns=final_features)
#%%
train_set = train[train.index <'2022-06-15 00:00:00']
# train_set = train[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
test_set = train[(train.index >= '2022-06-15 00:00:00')&(train.index <= '2022-06-30 23:59:59')]
train_target = target[train.index <'2022-06-15 00:00:00']
# train_target = target[(train.index >= '2022-04-01 00:00:00')&(train.index <= '2022-05-14 23:59:59')]
test_target = target[(train.index >= '2022-06-15 00:00:00')&(train.index <= '2022-06-30 23:59:59')]
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
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2022)
X_train, X_train_target = sm.fit_resample(X_train, X_train_target)
#%% GridSearchCV
# from sklearn.model_selection import GridSearchCV
# ## 定义参数取值范围
# learning_rate = [0.1, 0.3, 0.5, 0.7]
# feature_fraction = [0.3, 0.5, 0.8, 1]
# num_leaves = [16, 32, 64, 128]
# max_depth = [-1, 1, 3, 7, 10]
#
# parameters = {'learning_rate': learning_rate,
#               'feature_fraction':feature_fraction,
#               'num_leaves': num_leaves,
#               'max_depth': max_depth}
#
# model = LGBMClassifier(n_estimators=100)
# ## 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=3)
# clf = clf.fit(X_train,X_train_target)
# clf.best_params_## 网格搜索后的最优参数

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
import warnings
warnings.filterwarnings("ignore")
init_points = 20
n_iter = 100
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
#%%
LGB_BO.max['target']
#%%
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import lightgbm as lgb

# kf = StratifiedKFold(n_splits=10,random_state=20,shuffle=True)
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
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from numpy import sqrt,argmax
from sklearn.metrics import classification_report
from sklearn import metrics
fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, y_pred_train)
gmeans_train = sqrt(tpr_train * (1-fpr_train))
ix_train = argmax(gmeans_train)
print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))
thresholds_point_train = thresholds_train[ix_train]
yhat_train = [1 if y > thresholds_point_train else 0 for y in y_pred_train]
print("训练集表现：")
print(classification_report(yhat_train,X_train_target))
print(metrics.confusion_matrix(yhat_train, X_train_target))
#%% test set
# thresholds_point = 0.514
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
thresholds_point = thresholds_train[ix_train]
yhat = [1 if y > thresholds_point else 0 for y in y_pred]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
print(metrics.confusion_matrix(yhat, X_test_target))
#%%
auc = roc_auc_score(y_pred, X_test_target)
