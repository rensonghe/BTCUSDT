# -*- coding: utf-8 -*-
# Author:zq
import numpy as np
import pandas as pd


def test_data_consistency(test_data, col_name, ori_data, index):
    if test_data == 0:
        if ori_data[col_dict[col_name]] == 0:
            dif_rate = 0
        else:
            dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / ori_data[col_dict[col_name]])
    else:
        dif_rate = abs((test_data - ori_data[col_dict[col_name]]) / test_data)
    if dif_rate > 0.0001 and round(test_data, 6) != round(ori_data[col_dict[col_name]], 6):
        print('{}该数据批计算和实时流计算数值不一致---下标:{}---流数据:{}---批数据:{}'.format(col_name, index, test_data,
                                                                     ori_data[col_dict[col_name]]))


def shift_(interval):
    return -(interval + 1)


def diff_(df: bytearray):
    return df[-1] - df[-2]


def flow_data_1s_depth_wap5(df: bytearray, index, interval=60):
    feat_dict['bid_price1'] = np.append(feat_dict['bid_price1'], df[col_dict['bid_price1']])
    feat_dict['bid_size1'] = np.append(feat_dict['bid_size1'], df[col_dict['bid_size1']])
    feat_dict['ask_price1'] = np.append(feat_dict['ask_price1'], df[col_dict['ask_price1']])
    feat_dict['ask_size1'] = np.append(feat_dict['ask_size1'], df[col_dict['ask_size1']])
    feat_dict['bid_price2'] = np.append(feat_dict['bid_price2'], df[col_dict['bid_price2']])
    feat_dict['bid_size2'] = np.append(feat_dict['bid_size2'], df[col_dict['bid_size2']])
    feat_dict['ask_price2'] = np.append(feat_dict['ask_price2'], df[col_dict['ask_price2']])
    feat_dict['ask_size2'] = np.append(feat_dict['ask_size2'], df[col_dict['ask_size2']])

    # a = df['bid_size1'] * np.where(df['bid_price1'].diff() >= 0, 1, 0)
    # b = df['bid_size1'].shift() * np.where(df['bid_price1'].diff() <= 0, 1, 0)
    # c = df['ask_size1'] * np.where(df['ask_price1'].diff() <= 0, 1, 0)
    # d = df['ask_size1'].shift() * np.where(df['ask_price1'].diff() >= 0, 1, 0)
    # df['ofi'] = calc_ofi(df) TODO 未实现成功
    a = feat_dict['bid_size1'][-1] * (1 if -diff_(feat_dict['bid_price1']) >= 0 else 0)
    b = feat_dict['bid_size1'][shift_(1)] * (1 if -diff_(feat_dict['bid_price1']) <= 0 else 0)
    c = feat_dict['ask_size1'][-1] * (1 if -diff_(feat_dict['ask_price1']) <= 0 else 0)
    d = feat_dict['ask_size1'][shift_(1)] * (1 if -diff_(feat_dict['ask_price1']) >= 0 else 0)
    ofi = a - b - c + d
    feat_dict['ofi'] = np.append(
        feat_dict['ofi'],
        ofi)
    test_data_consistency(feat_dict['ofi'][-1],
                          'ofi', df, index)


if __name__ == '__main__':
    import time

    data_agg = pd.read_csv('depth_1s_wap5.csv')
    feat_dict = {i: np.array([]) for i in (list(data_agg.columns.to_list()) + ['wap1', 'wap2', 'wap3', 'wap4'])}

    col_dict = {}  # 将列表中的列名的下标作为值 列名做为键
    for i, value in enumerate(data_agg.columns.to_list() + ['wap1', 'wap2', 'wap3', 'wap4']):
        # print(index, value)
        col_dict[value] = i
    agg_values = data_agg.values
    t1 = time.time()
    for i in range(1000):
        # flow_data_1s_trade_vwap(agg_values[i], i)
        # flow_data_1s_trade_base(agg_values[i], i)
        # flow_data_1s_depth_wap1(agg_values[i], i)
        # flow_data_1s_depth_wap2(agg_values[i], i)
        # flow_data_1s_depth_wap3(agg_values[i], i)
        # flow_data_1s_depth_wap4(agg_values[i], i)
        flow_data_1s_depth_wap5(agg_values[i], i)
    print(time.time() - t1)
