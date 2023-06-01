#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
minio = fs.S3FileSystem(endpoint_override="192.168.34.40:9000", access_key="ozWQ1ifSjatFzDkh",
                        secret_key="TYaQNlrcJhnPswmtpHjYPBz76mAaUluG", scheme="http")

year = 2022
month = 4
def get_data_from_minio(platform, symbol, year, month, dir_name, data_type, start_time=None, end_time=None):
    # 注意要看minio中的数据来做过滤
    filters = [("symbol", "=", symbol), ("year", "=", year), ("month", "=", month)]
    if start_time:
        filters.append(('closetime', '>=', start_time))
    if end_time:
        filters.append(('closetime', '<=', end_time))
    dataset = pq.ParquetDataset('{}/{}/'.format(dir_name, data_type), filters=filters, filesystem=minio)
    return dataset.read_pandas().to_pandas()
tick_1s = get_data_from_minio('gate_swap_u', 'btcusdt', year, month, 'datafile/feat', 'trade_1s_vwap')

for data_type in ['trade_1s_base', 'depth_1s_wap1', 'depth_1s_wap2', 'depth_1s_wap3',
                  'depth_1s_wap4', 'depth_1s_wap5', 'depth_1s_price', 'depth_1s_size']:
    new_data = get_data_from_minio('gate_swap_u', 'btcusdt', year, month, 'datafile/feat', data_type)
    new_columns_list = new_data.columns.to_list()
    for columns_name in tick_1s.columns.to_list():
        # 保留closetime这一列
        if columns_name in new_columns_list and columns_name != 'closetime':
            tick_1s.drop([columns_name], axis=1, inplace=True)
    tick_1s = tick_1s.merge(new_data, how='left', on='closetime')
tick_1s = tick_1s.fillna(method='ffill')
tick_1s = tick_1s.fillna(method='bfill')
tick_1s = tick_1s.replace(np.inf, 1)
tick_1s = tick_1s.replace(-np.inf, -1)
tick_1s['datetime'] = pd.to_datetime(tick_1s['closetime']+28800, unit='s')
tick_1s = tick_1s.set_index('datetime')
tick_1s = tick_1s.drop(['symbol', 'year', 'month'], axis=1)
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
tick_1s = reduce_mem_usage(tick_1s)[0]

tick_1s.to_csv('/songhe/AI1.0/data/tick_1s_{}_{}.csv'.format(year, month), index=False)

