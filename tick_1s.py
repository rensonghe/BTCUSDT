import numpy as np
import pandas as pd
from functools import reduce
import pyarrow
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
# month = 1
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
from tqdm import tqdm
#%%
all_data = pd.DataFrame()
symbol = 'btcusdt'
platform = 'gate_swap_u'
year = 2022
for month in tqdm(range(6, 7)):
    # 拿orderbook+trade的数据
    # data_type = ''
    filters = [('symbol', '=', symbol), ('year', '=', year), ('month','=',month)]
    dataset = pq.ParquetDataset('datafile/tick/tick_1s/gate_swap_u', filters=filters, filesystem=minio)
    tick_1s = dataset.read_pandas().to_pandas()

#%%
tick_1s['datetime'] = pd.to_datetime(tick_1s['closetime']+28800, unit='s')