# %%
import time
from pyarrow import fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

#%%  计算这一行基于bid和ask的wap
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

#%%
# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

# def realized_quarticity(series):
#     return (np.sum(series**4)*series.shape[0]/3)
#
# def reciprocal_transformation(series):
#     return np.sqrt(1/series)*100000
#
# def square_root_translation(series):
#     return series**(1/2)
#%%  计算订单簿因子，数据间隔为每秒
def book_preprocessor(data):

    df = data

    rolling = 60

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    # df['wap1_quarticity']=realized_quarticity(df['wap1'])
    # df['wap1_reciprocal'] = reciprocal_transformation(df['wap1'])
    # df['wap1_square_root'] = square_root_translation(df['wap1'])
    # df['wap2_quarticity'] = realized_quarticity(df['wap2'])
    # df['wap2_reciprocal'] = reciprocal_transformation(df['wap2'])
    # df['wap2_square_root'] = square_root_translation(df['wap2'])
    # df['wap3_quarticity']=realized_quarticity(df['wap3'])
    # df['wap3_reciprocal'] = reciprocal_transformation(df['wap3'])
    # df['wap3_square_root'] = square_root_translation(df['wap3'])
    # df['wap4_quarticity'] = realized_quarticity(df['wap4'])
    # df['wap4_reciprocal'] = reciprocal_transformation(df['wap4'])
    # df['wap4_square_root'] = square_root_translation(df['wap4'])

    df['wap1_shift2'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['wap1_shift60'] = df['wap1'].shift(1) - df['wap1'].shift(rolling)
    df['wap1_shift120'] = df['wap1'].shift(1) - df['wap1'].shift(rolling*2)
    df['wap1_shift300'] = df['wap1'].shift(1) - df['wap1'].shift(rolling * 5)

    df['wap2_shift2'] = df['wap2'].shift(1) - df['wap2'].shift(2)
    df['wap2_shift60'] = df['wap2'].shift(1) - df['wap2'].shift(rolling)
    df['wap2_shift120'] = df['wap2'].shift(1) - df['wap2'].shift(rolling*2)
    df['wap2_shift300'] = df['wap2'].shift(1) - df['wap2'].shift(rolling * 5)

    df['wap3_shift2'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['wap3_shift60'] = df['wap3'].shift(1) - df['wap3'].shift(rolling)
    df['wap3_shift120'] = df['wap3'].shift(1) - df['wap3'].shift(rolling*2)
    df['wap3_shift300'] = df['wap3'].shift(1) - df['wap3'].shift(rolling * 5)

    df['wap4_shift2'] = df['wap4'].shift(1) - df['wap4'].shift(2)
    df['wap4_shift60'] = df['wap4'].shift(1) - df['wap4'].shift(rolling)
    df['wap4_shift120'] = df['wap4'].shift(1) - df['wap4'].shift(rolling*2)
    df['wap4_shift300'] = df['wap4'].shift(1) - df['wap4'].shift(rolling * 5)

    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df['ask_price1'] == df['lask_price1'].shift(1), df['ask_size1'] - df['ask_size1'].shift(1), 0)
    df['vtA'] = np.where(df['ask_price1'] > df['ask_price1'].shift(1), df['ask_size1'], df['pre_vtA'])
    df['pre_vtB'] = np.where(df['bid_price1'] == df['bid_price1'].shift(1), df['bid_size1'] - df['bid_size1'].shift(1), 0)
    df['vtB'] = np.where(df['bid_price1'] > df['bid_price1'].shift(1), df['bid_size1'], df['pre_vtB'])

    df['Oiab'] = df['vtB']-df['vtA']

    df['bid_ask_size1_minus'] = df['bid_size1']-df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1']+df['ask_size1']
    df['bid_ask_size2_minus'] = df['bid_size2'] - df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2'] + df['ask_size2']
    df['bid_ask_size3_minus'] = df['bid_size3'] - df['ask_size3']
    df['bid_ask_size3_plus'] = df['bid_size3'] + df['ask_size3']
    df['bid_ask_size4_minus'] = df['bid_size4'] - df['ask_size4']
    df['bid_ask_size4_plus'] = df['bid_size4'] + df['ask_size4']

    df['bid_size1_shift'] = df['bid_size1']-df['bid_size1'].shift()
    df['ask_size1_shift'] = df['ask_size1']-df['ask_size1'].shift()
    df['bid_size2_shift'] = df['bid_size2'] - df['bid_size2'].shift()
    df['ask_size2_shift'] = df['ask_size2'] - df['ask_size2'].shift()
    df['bid_size3_shift'] = df['bid_size3'] - df['bid_size3'].shift()
    df['ask_size3_shift'] = df['ask_size3'] - df['ask_size3'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus']/df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_ask_size3_spread'] = df['bid_ask_size3_minus'] / df['bid_ask_size3_plus']
    df['bid_ask_size4_spread'] = df['bid_ask_size4_minus'] / df['bid_ask_size4_plus']

    df['roliing_mid_price1_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['rolling_mid_price1_std'] = df['mid_price1'].rolling(rolling).std()

    df['rolling_HR1_mean'] = df['HR1'].rolling(rolling).mean()

    df['rolling_bid_ask_size1_minus_mean1'] = df['bid_ask_size1_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size2_minus_mean1'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size3_minus_mean1'] = df['bid_ask_size3_minus'].rolling(rolling).mean()


    df['rolling_bid_size1_shift_mean1'] = df['bid_size1_shift'].rolling(rolling).mean()
    df['rolling_bid_size1_shift_mean3'] = df['bid_size1_shift'].rolling(2*rolling).mean()
    df['rolling_bid_size1_shift_mean5'] = df['bid_size1_shift'].rolling(5*rolling).mean()
    df['rolling_ask_size1_shift_mean1'] = df['ask_size1_shift'].rolling(rolling).mean()
    df['rolling_ask_size1_shift_mean3'] = df['ask_size1_shift'].rolling(2*rolling).mean()
    df['rolling_ask_size1_shift_mean5'] = df['ask_size1_shift'].rolling(5*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean2'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_spread_mean3'] = df['bid_ask_size1_spread'].rolling(2*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean5'] = df['bid_ask_size1_spread'].rolling(5*rolling).mean()


    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return2'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(2)) * 100
    df['log_return3'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(2)) * 100
    df['log_return4'] = np.log(df['wap4'].shift(1)/df['wap4'].shift(2))*100


    df['log_return_wap1_shift60'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(rolling))*100
    df['log_return_wap2_shift60'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling)) * 100
    df['log_return_wap3_shift60'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling)) * 100
    df['log_return_wap4_shift60'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling)) * 100

    df['log_return_wap1_shift120'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(rolling*2))*100
    df['log_return_wap2_shift120'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling*2)) * 100
    df['log_return_wap3_shift120'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling*2)) * 100
    df['log_return_wap4_shift120'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling*2)) * 100


    df['ewm_wap1_mean'] = pd.DataFrame.ewm(df['wap1'],span=rolling).mean()
    df['ewm_wap2_mean'] = pd.DataFrame.ewm(df['wap2'], span=rolling).mean()
    df['ewm_wap3_mean'] = pd.DataFrame.ewm(df['wap3'], span=rolling).mean()
    df['ewm_wap4_mean'] = pd.DataFrame.ewm(df['wap4'], span=rolling).mean()


    df['rolling_mean1'] = df['wap1'].rolling(rolling).mean()
    df['rolling_std1'] = df['wap1'].rolling(rolling).std()
    df['rolling_min1'] = df['wap1'].rolling(rolling).min()
    df['rolling_max1'] = df['wap1'].rolling(rolling).max()
    df['rolling_skew1'] = df['wap1'].rolling(rolling).skew()
    df['rolling_kurt1'] = df['wap1'].rolling(rolling).kurt()
    df['rolling_quantile1_25'] = df['wap1'].rolling(rolling).quantile(.25)
    df['rolling_quantile1_75'] = df['wap1'].rolling(rolling).quantile(.75)

    df['rolling_mean2'] = df['wap2'].rolling(rolling).mean()
    df['rolling_std2'] = df['wap2'].rolling(rolling).std()
    df['rolling_min2'] = df['wap2'].rolling(rolling).min()
    df['rolling_max2'] = df['wap2'].rolling(rolling).max()
    df['rolling_skew2'] = df['wap2'].rolling(rolling).skew()
    df['rolling_kurt2'] = df['wap2'].rolling(rolling).kurt()
    df['rolling_quantile2_25'] = df['wap2'].rolling(rolling).quantile(.25)
    df['rolling_quantile2_75'] = df['wap2'].rolling(rolling).quantile(.75)


    df['rolling_mean3'] = df['wap3'].rolling(rolling).mean()
    df['rolling_var3'] = df['wap3'].rolling(rolling).var()
    df['rolling_min3'] = df['wap3'].rolling(rolling).min()
    df['rolling_max3'] = df['wap3'].rolling(rolling).max()
    df['rolling_skew3'] = df['wap3'].rolling(rolling).skew()
    df['rolling_kurt3'] = df['wap3'].rolling(rolling).kurt()
    df['rolling_median3'] = df['wap3'].rolling(rolling).median()
    df['rolling_quantile3_25'] = df['wap3'].rolling(rolling).quantile(.25)
    df['rolling_quantile3_75'] = df['wap3'].rolling(rolling).quantile(.75)


    df['rolling_mean4'] = df['wap4'].rolling(rolling).mean()
    df['rolling_std4'] = df['wap4'].rolling(rolling).std()
    df['rolling_min4'] = df['wap4'].rolling(rolling).min()
    df['rolling_max4'] = df['wap4'].rolling(rolling).max()
    df['rolling_skew4'] = df['wap4'].rolling(rolling).skew()
    df['rolling_kurt4'] = df['wap4'].rolling(rolling).kurt()
    df['rolling_median4'] = df['wap4'].rolling(rolling).median()
    df['rolling_quantile4_25'] = df['wap4'].rolling(rolling).quantile(.25)
    df['rolling_quantile4_75'] = df['wap4'].rolling(rolling).quantile(.75)


    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])
    df['wap_balance2'] = abs(df['wap1'] - df['wap3'])
    df['wap_balance3'] = abs(df['wap2'] - df['wap3'])
    df['wap_balance4'] = abs(df['wap3'] - df['wap4'])

    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['price_spread3'] = (df['ask_price3'] - df['bid_price3']) / ((df['ask_price3'] + df['bid_price3']) / 2)
    df['price_spread4'] = (df['ask_price4'] - df['bid_price4']) / ((df['ask_price4'] + df['bid_price4']) / 2)

    # print(df.columns)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)


    return df
#%% 计算订单流因子，数据间隔为每秒
def trade_preprocessor(data):
    df = data
    # df['log_return'] = np.log(df['last_price']).shift()

    df['amount'] = df['last_price'] * df['size']

    rolling = 60

    df['mid_price'] = np.where(df['size'] > 0, (df['amount'] - df['amount'].shift(1)) / df['size'], df['last_price'])
    df['rolling_mid_price_mean60'] = df['mid_price'].rolling(rolling).mean()
    df['rolling_mid_price_mean120'] = df['mid_price'].rolling(rolling*2).mean()
    df['rolling_mid_price_std60'] = df['mid_price'].rolling(rolling).std()
    df['rolling_mid_price_std120'] = df['mid_price'].rolling(rolling*2).std()

    df['last_price_shift1'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift60'] = df['last_price'].shift(1) - df['last_price'].shift(rolling)
    df['last_price_shift120'] = df['last_price'].shift(1) - df['last_price'].shift(rolling*2)
    df['last_price_shift300'] = df['last_price'].shift(1) - df['last_price'].shift(rolling*5)


    df['log_return_last_price_shift1'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(1)) * 100
    df['log_return_last_price_shift60'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(rolling)) * 100
    df['log_return_last_price_shift120'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(rolling*2)) * 100
    df['log_return_last_price_shift300'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(rolling*5)) * 100


    df['rolling_mean_size'] = df['size'].rolling(rolling).mean()
    df['rolling_var_size'] = df['size'].rolling(rolling).var()
    df['rolling_std_size'] = df['size'].rolling(rolling).std()
    df['rolling_sum_size'] = df['size'].rolling(rolling).sum()
    df['rolling_min_size'] = df['size'].rolling(rolling).min()
    df['rolling_max_size'] = df['size'].rolling(rolling).max()
    df['rolling_skew_size'] = df['size'].rolling(rolling).skew()
    df['rolling_kurt_size'] = df['size'].rolling(rolling).kurt()
    df['rolling_median_size'] = df['size'].rolling(rolling).median()

    df['ewm_mean_size'] = pd.DataFrame.ewm(df['size'], span=rolling).mean()
    df['ewm_std_size'] = pd.DataFrame.ewm(df['size'], span=rolling).std()

    df['size_percentile_25'] = df['size'].rolling(rolling).quantile(.25)
    df['size_percentile_25_120'] = df['size'].rolling(rolling*2).quantile(.25)
    df['size_percentile_25_300'] = df['size'].rolling(rolling*5).quantile(.25)
    df['size_percentile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['size_percentile_75_120'] = df['size'].rolling(rolling*2).quantile(.75)
    df['size_percentile_75_300'] = df['size'].rolling(rolling*2).quantile(.75)

    df['size_percentile_60'] = df['size_percentile_75'] - df['size_percentile_25']
    df['size_percentile_120'] = df['size_percentile_75_120'] - df['size_percentile_25_120']
    df['size_percentile_300'] = df['size_percentile_75_300'] - df['size_percentile_25_300']


    df['price_percentile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['price_percentile_25_120'] = df['last_price'].rolling(rolling*2).quantile(.25)
    df['price_percentile_25_300'] = df['last_price'].rolling(rolling*5).quantile(.25)
    df['price_percentile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['price_percentile_75_120'] = df['last_price'].rolling(rolling*2).quantile(.75)
    df['price_percentile_75_300'] = df['last_price'].rolling(rolling*5).quantile(.75)

    df['price_percentile'] = df['price_percentile_75'] - df['price_percentile_25']
    df['price_percentile'] = df['price_percentile_75_120'] - df['price_percentile_25_120']
    df['price_percentile'] = df['price_percentile_75_300'] - df['price_percentile_25_300']


    df['rolling_mean_amount'] = df['amount'].rolling(rolling).mean()
    df['rolling_mean_amount_120'] = df['amount'].rolling(rolling*2).mean()
    df['rolling_mean_amount_300'] = df['amount'].rolling(rolling*5).mean()
    df['rolling_quantile_25_amount'] = df['amount'].rolling(rolling).quantile(.25)
    df['rolling_quantile_25_amount_120'] = df['amount'].rolling(rolling*2).quantile(.25)
    df['rolling_quantile_25_amount_300'] = df['amount'].rolling(rolling*5).quantile(.25)
    df['rolling_quantile_75_amount'] = df['amount'].rolling(rolling).quantile(.75)
    df['rolling_quantile_75_amount_120'] = df['amount'].rolling(rolling*2).quantile(.75)
    df['rolling_quantile_75_amount_300'] = df['amount'].rolling(rolling*5).quantile(.75)

    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()

    # print(df.columns)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df



def depth4(df):
    out = pd.DataFrame()
    for symbol in df['symbol'].unique():
        cur2 = df.loc[df['symbol']==symbol]
        tmp = pd.DataFrame()
        cur2['rid'] = cur2['ask'].groupby(cur2['update']).rank()
        cur2 = cur2.loc[cur2['rid']<=4]
        cur21 = cur2.pivot(index='update',columns='rid',values='ask')    
        cur21 = cur21.rename(columns={1: "ask_price1", 2: "ask_price2",3: "ask_price3",4: "ask_price4"})
        cur22 = cur2.pivot(index='update',columns='rid',values='ask_num')
        cur22 = cur22.rename(columns={1: "ask_size1", 2: "ask_size2",3: "ask_size3",4: "ask_size4"})
        cur23 = cur2.pivot(index='update',columns='rid',values='bid')
        cur23 = cur23.rename(columns={1: "bid_price1", 2: "bid_price2",3: "bid_price3",4: "bid_price4"})
        cur24 = cur2.pivot(index='update',columns='rid',values='bid_num')
        cur24 = cur24.rename(columns={1: "bid_size1", 2: "bid_size2",3: "bid_size3",4: "bid_size4"})
        cur2 = pd.concat([cur21, cur22, cur23,cur24], axis = 1)
        cur2.reset_index(inplace=True)
        cur2 = cur2.sort_values('update')
        tmp = pd.concat([tmp, cur2], axis = 0)
        tmp['symbol'] = symbol
        out = pd.concat([out, tmp], axis = 0)
    return out 
          

# %%
import time
from pyarrow import fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
minio = fs.S3FileSystem(endpoint_override="192.168.34.40:9000", access_key="ozWQ1ifSjatFzDkh", secret_key="TYaQNlrcJhnPswmtpHjYPBz76mAaUluG", scheme="http")
# 从minio 中拿数据
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
# month = 2
for month in range(6, 7):
    # 拿orderbook的数据
    data_type = 'orderbook'
    filters = [('symbol', '=', symbol) and ('platform', '=', platform) and ('year', '=', year) and ('month','=',month)]
    dataset = pq.ParquetDataset('datafile/tick/{}'.format(data_type),filters=filters,filesystem=minio)
    depth = dataset.read_pandas().to_pandas()
    # 过滤出btc 这个币种的数据
    depth = depth[depth.symbol == symbol]
    # 将深度的行数据转换成列数据
    depth = depth4(depth)
    # 替换列名
    depth.rename(columns={'update': 'datetime'}, inplace=True)
    # 删除不需要的列
    depth = depth.drop(['symbol'], axis=1)
    depth[0:10000].to_csv('{}_ori_depth_test{}{}.csv'.format(symbol, year, month), index=False)
    depth.to_csv('{}_ori_depth{}{}.csv'.format(symbol, year, month), index=False)

    # 拿orderflow的数据
    data_type = 'trade'
    filters = [('symbol', '=', symbol) and ('platform', '=', platform) and ('year', '=', year) and ('month','=',month)]
    dataset_trade = pq.ParquetDataset('datafile/tick/{}'.format(data_type),filters=filters,filesystem=minio)
    trade = dataset_trade.read_pandas().to_pandas()
    # 过滤出btc 这个币种的数据
    trade = trade[trade.symbol == symbol]
    # 根据 timestamp 升序排列
    trade = trade.sort_values(by='timestamp', ascending=True)
    # 替换列名
    trade.rename(columns={'timestamp': 'datetime', 'price': 'last_price'}, inplace=True)
    # 删除不需要的列
    trade = trade.drop(['symbol', 'dealid', 'year', 'month'], axis=1)
    trade['size'] = abs(trade['size'])
    # 时间戳取整
    # trade['datetime'] = trade['datetime'] // 1
    trade[0:10000].to_csv('{}_ori_trade_test{}{}.csv'.format(symbol, year, month), index=False)
    trade.to_csv('{}_ori_trade{}{}.csv'.format(symbol, year, month), index=False)

# %%
import time
t1 = time.time()
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
# month = 1
for month in range(6, 7):
    trade_data = pd.read_csv('{}_ori_trade{}{}.csv'.format(symbol, year, month))
    trade_data = trade_preprocessor(trade_data)
    # trade_data
    # 将13位时间戳向下取整成10位时间戳 默认是utc-0 转成utc-8时区 这一步是可以做加速的 TODO   
    trade_data['datetime'] = pd.to_datetime(trade_data['datetime'] + 28800, unit='s')
    trade_data[0:10000].to_csv('{}_trade_data_test{}{}.csv'.format(symbol, year, month), index=False)
    # 这一步是可以做加速的 TODO
    def get_vwap(data):
        v = data['size']
        p = data['last_price']
        data['last_price_vwap'] = np.sum(p*v) / np.sum(v)
        return data
    print(time.time() - t1)
    # 通过trade产出的因子将数据进项聚合为1分钟的数据
    time_group = trade_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply(get_vwap)
    time_group_trade = time_group.groupby(pd.Grouper(freq='1min')).agg(np.mean)
    time_group_trade = time_group_trade.dropna(axis=0,how='all')
    time_group_trade = time_group_trade.reset_index()    #time_group为最终传入模型的因子
    time_group_trade.to_csv('{}_time_group_trade{}{}.csv'.format(symbol, year, month), index=False)
    print(time.time() - t1)

# %%
import time
t1 = time.time()
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
# month = 1
for month in range(6, 7):
    book_data = pd.read_csv('{}_ori_depth{}{}.csv'.format(symbol, year, month))
    book_data = book_preprocessor(book_data)
    # # 将13位时间戳向下取整成10位时间戳 转成+8时区 这一步是可以做加速的 TODO
    book_data['datetime'] = pd.to_datetime(book_data['datetime'] + 28800, unit='s')
    book_data[0:10000].to_csv('{}_book_data_test{}{}.csv'.format(symbol, year, month), index=False)
    print(time.time() - t1)
    # #%% 通过book产出的因子聚合为1分钟的数据
    time_group_book = book_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg(np.mean)
    time_group_book = time_group_book.dropna(axis=0,how='all')
    time_group_book = time_group_book.reset_index()
    time_group_book.to_csv('{}_time_group_book{}{}.csv'.format(symbol, year, month), index=False)
    print(time.time() - t1)

# %%
ori_trade = pd.read_csv('t_trade.csv')
trade_data = trade_preprocessor(ori_trade)
# trade_data
# 将13位时间戳向下取整成10位时间戳 默认是utc-0 转成utc-8时区 这一步是可以做加速的 TODO   
trade_data['datetime'] = pd.to_datetime(trade_data['datetime'] // 1000 + 28800, unit='s')
# trade_data.to_csv('trade_data{}{}.csv'.format(year, month), index=False)
# 这一步是可以做加速的 TODO
def get_vwap(data):
    v = data['size']
    p = data['last_price']
    data['last_price_vwap'] = np.sum(p*v) / np.sum(v)
    return data
print(time.time() - t1)
# 通过trade产出的因子将数据进项聚合为1分钟的数据
time_group = trade_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply(get_vwap)
time_group_trade = time_group.groupby(pd.Grouper(freq='1min')).agg(np.mean)
time_group_trade = time_group_trade.dropna(axis=0,how='all')
time_group_trade = time_group_trade.reset_index()    #time_group为最终传入模型的因子
time_group_trade.to_csv('t_time_group_trade.csv', index=False)

# %%
ori_depth = pd.read_csv('t_depth.csv')
book_data = book_preprocessor(ori_depth)
# # 将13位时间戳向下取整成10位时间戳 转成+8时区 这一步是可以做加速的 TODO
book_data['datetime'] = pd.to_datetime(book_data['datetime'] // 1000 + 28800, unit='s')
# book_data.to_csv('book_data{}{}.csv'.format(year, month), index=False)
print(time.time() - t1)
# #%% 通过book产出的因子聚合为1分钟的数据
time_group_book = book_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg(np.mean)
time_group_book = time_group_book.dropna(axis=0,how='all')
time_group_book = time_group_book.reset_index()
time_group_book.to_csv('t_time_group_book.csv', index=False)
print(time.time() - t1)

# %%
bu_ori_trade = pd.read_csv('ori_trade_1654012800_1654092068.csv')
bu_ori_trade = bu_ori_trade.loc[bu_ori_trade['Datetime'] >= 1654012800]['Datetime']
bu_ori_trade
# trade_ori = pd.read_csv('BTC_USDT-202206.csv', header=None)
# trade_ori[0][9884974]

# trade_ori[3] = abs(trade_ori[3])
# trade_ori.rename(columns={0:s 'datetime', 2:'last_price', 3: 'size'}, inplace=True)
# trade_ori = trade_ori.drop([1], axis=1)
# # 根据 timestamp 升序排列
# trade_ori = trade_ori.sort_values(by='datetime', ascending=True)
# trade_ori = trade_ori[0:100]
# trade_ori.to_csv('ori_trade20226.csv', index=False)


