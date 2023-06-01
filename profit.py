import time

import numpy as np
import random
from pprint import pprint

from data_analysis.bts.utils.tool_util import get_float_value_decimal_lenth
from data_analysis.cta.index_old import array_time

class Profit(object):

    def __init__(self, symbols: list, start_fund=1000):
        self.principal = {symbol: start_fund for symbol in symbols}  # 本金
        self.taker_fee_rate = 0.0003  # 手续费
        self.slippage = 0.0003  # 滑点
        self.profit_interval = 86400  # 利润区间 10位时间戳1天时间段
        self.sqrt_base_hour = 3600    # 10位时间戳1小时时间段
        self.timestamps_initialization = {symbol: False for symbol in symbols}  # 是否初始化时间戳
        """利润统计模块： 持仓, 利润"""
        self.position = {symbol: {'price': 0, 'amount': 0, 'number': 0, 'volume': 0, 'timestamp': 0} for symbol in
                         symbols}
        self.profit_data = {
            symbol: {"profit_interval": self.profit_interval, 'max_profit': 0, 'max_back': 0, 'total_profit': 0,
                     'profit_list': list(),
                     'count_profit': {'buy_time': 0, 'sell_time': 0, 'buy_+': 0, 'sell_+': 0, 'buy_-': 0,
                                      'sell_-': 0, 'total': 0, 'buy_+_time': 0, 'sell_+_time': 0, 'buy_-_time': 0,
                                      'sell_-_time': 0},
                     'timestamps': {'start_time': 0, 'now_time': 0, 'profit_interval_time': 0}
                     } for symbol in symbols
        }
        self.profit_list_time = {symbol:list() for symbol in symbols}
        self.profit_list_times = {symbol:list() for symbol in symbols}
        self.profitAndTime = {symbol:list() for symbol in symbols}

    def setProfitinterval(self, symbol, profit_interval: int):
        """
            根据symbol设置利润区间
        """
        self.profit_data[symbol]['profit_interval'] = profit_interval

    def profit_every_order(self, order_list):
        try:
            for order in order_list:
                symbol = order['symbol']
                if order['side'] == 'sell':
                    if self.position[symbol]['amount'] > 0:
                        profit = (order['price'] - self.position[symbol]['price']) * min(self.position[symbol]['amount'], order['size'])
                        if profit >= 0:
                            self.profit_data[symbol]['count_profit']['buy_+'] += profit
                            self.profit_data[symbol]['count_profit']['buy_+_time'] += 1
                        else:
                            self.profit_data[symbol]['count_profit']['buy_-'] += profit
                            self.profit_data[symbol]['count_profit']['buy_-_time'] += 1
                        self.profit_data[symbol]['count_profit']['buy_time'] += 1  # 平仓的话是做多+1
                        # 更新持仓
                        update_amount = round(self.position[symbol]['amount'] - order['size'],
                                              get_float_value_decimal_lenth(order['size']))
                        if update_amount < 0:  # 多仓转空仓
                            self.position[symbol]['price'] = round(order['price'], 6)
                        elif update_amount == 0:
                            self.position[symbol]['price'] = 0
                        # 只平仓且仓位未平完不会影响开仓均价
                        self.position[symbol]['amount'] = update_amount
                        self.position[symbol]['number'] += order['size']
                        self.profit_build(order)
                        self.profitAndTime[symbol].append([profit, order['timestamp']])
                    else:
                        # 持仓价格为正,数量为负
                        self.position[symbol]['price'] = round(
                            (order['price'] * order['size'] + self.position[symbol]['price'] * abs(
                                self.position[symbol]['amount'])) / (
                                    abs(self.position[symbol]['amount']) + order['size']), 6)
                        self.position[symbol]['volume'] += order['price'] * order['size']
                        self.position[symbol]['number'] += order['size']
                        self.position[symbol]['amount'] -= order['size']
                else:
                    if self.position[symbol]['amount'] < 0:
                        profit = (self.position[symbol]['price'] - order['price']) * min(-self.position[symbol]['amount'], order['size'])
                        if profit >= 0:
                            self.profit_data[symbol]['count_profit']['sell_+'] += profit
                            self.profit_data[symbol]['count_profit']['sell_+_time'] += 1
                        else:
                            self.profit_data[symbol]['count_profit']['sell_-'] += profit
                            self.profit_data[symbol]['count_profit']['sell_-_time'] += 1
                        self.profit_data[symbol]['count_profit']['sell_time'] += 1  # 平仓的话是做多+1
                        # 更新持仓
                        update_amount = round(self.position[symbol]['amount'] + order['size'],
                                              get_float_value_decimal_lenth(order['size']))
                        if update_amount > 0:  # 空仓转多仓
                            self.position[symbol]['price'] = round(order['price'], 6)
                        elif update_amount == 0:
                            self.position[symbol]['price'] = 0
                        # 只平仓且仓位未平完不会影响开仓均价
                        self.position[symbol]['amount'] = update_amount
                        self.position[symbol]['number'] += order['size']
                        self.profit_build(order)
                        self.profitAndTime[symbol].append([profit, order['timestamp']])
                    else:
                        self.position[symbol]['price'] = round(
                            (order['price'] * order['size'] + self.position[symbol]['price'] * self.position[symbol][
                                'amount']) /
                            (order['size'] + self.position[symbol]['amount']), 6)
                        self.position[symbol]['volume'] += order['price'] * order['size']
                        self.position[symbol]['number'] += order['size']
                        self.position[symbol]['amount'] += order['size']

        except Exception as e:
            print("profit_every_trade_self错误原因:{},行数:{}".format(e, e.__traceback__.tb_lineno))

    def profit_build(self, order):
        symbol = order['symbol']
        self.profit_data[symbol]['count_profit']['total'] = self.profit_data[symbol]['count_profit']['buy_+'] + self.profit_data[symbol]['count_profit']['buy_-'] + \
                                                            self.profit_data[symbol]['count_profit']['sell_+'] + self.profit_data[symbol]['count_profit']['sell_-']
        self.profit_data[symbol]['total_profit'] = self.profit_data[symbol]['count_profit']['total'] #总收益
        if self.profit_data[symbol]['max_profit'] < self.profit_data[symbol]['count_profit']['total']:
            self.profit_data[symbol]['max_profit'] = self.profit_data[symbol]['count_profit']['total']
        max_back = round(self.profit_data[symbol]['max_profit'] - self.profit_data[symbol]['count_profit']['total'],4)
        if max_back > self.profit_data[symbol]['max_back']:
            self.profit_data[symbol]['max_back'] = max_back

        # 如果没有初始化，则初始化时间，初始化了之后，判断根据时间判断往利润列表里添加利润
        if self.timestamps_initialization[symbol] is False:
            for k in self.profit_data[symbol]['timestamps']:
                self.profit_data[symbol]['timestamps'][k] = order['timestamp']
            self.timestamps_initialization[symbol] = True
            if len(str(order['timestamp'])) == 13:  # 当时间戳为13位时
                self.profit_data[symbol]['profit_interval'] = self.profit_data[symbol]['profit_interval'] * 1000
                self.sqrt_base_hour = self.sqrt_base_hour * 1000
        else:
            self.profit_data[symbol]['timestamps']['now_time'] = order['timestamp']
            if self.profit_data[symbol]['timestamps']['now_time'] - \
                    self.profit_data[symbol]['timestamps']['profit_interval_time'] > \
                    self.profit_data[symbol]['profit_interval']:
                for i in range((self.profit_data[symbol]['timestamps']['now_time'] -
                                self.profit_data[symbol]['timestamps']['profit_interval_time']) //
                               self.profit_data[symbol]['profit_interval']):
                    self.profit_data[symbol]['profit_list'].append(
                        round(self.profit_data[symbol]['count_profit']['total'], 6))
                    self.profit_list_time[symbol].append(array_time(self.profit_data[symbol]['timestamps']['profit_interval_time'], flag=1))
                    self.profit_list_times[symbol].append(self.profit_data[symbol]['timestamps']['profit_interval_time'])
                self.profit_data[symbol]['timestamps']['profit_interval_time'] = order['timestamp'] // self.profit_data[symbol]['profit_interval'] * self.profit_data[symbol]['profit_interval']

        self.position[symbol]['timestamp'] = order['timestamp']

    def generate_profit_statistics(self, symbol, print_flag=False):
        try:
            sqrt_base = int(self.sqrt_base_hour / self.profit_data[symbol]['profit_interval'] * 24 * 365)
            year_profit_info = self.generate_profit_statistics_year_profit(symbol)
            volatility = self.generate_profit_statistics_volatility(symbol, sqrt_base)
            sharpe_ratio = self.generate_profit_statistics_sharpe_ratio(symbol, sqrt_base,
                                                                        year_profit_info['year_profit_rate'])
            karma_ratio = self.generate_profit_statistics_karma_ratio(symbol)
            max_back_ratio = self.generate_profit_statistics_max_back(symbol)
            win_info = self.generate_profit_total_times(symbol)
            # 生成利润统计报表
            if print_flag:
                pprint(self.position)
                pprint(self.profit_data)
                print("预计结果:")
                print("年化收益:{}".format(str(year_profit_info['year_profit'])))
                print("年化收益率:{}%".format(year_profit_info['year_profit_rate']))
                print("总交易次数:{}, 胜率:{}%".format(win_info['total_times'], win_info['wrate']))
                print("最大回撤:{}%".format(max_back_ratio))
                # print("最大回撤:{}%".format(round((self.profit_data[symbol]['max_back'] / self.principal[symbol]
                # if self.profit_data[symbol]['max_profit'] == 0 else self.profit_data[symbol]['max_profit']) * 100,6)))
                print("利润波动率:{}".format(volatility))
                print("夏普比率:{}".format(sharpe_ratio))
                print("卡玛比率:{}".format(karma_ratio))
            return {'year_profit': round(year_profit_info['year_profit'], 2),
                    'year_profit_rate': round(year_profit_info['year_profit_rate'], 2),
                    'total_times': win_info['total_times'],
                    'win_rate': round(win_info['wrate'], 2),
                    'max_back': round(max_back_ratio, 2),
                    'volatility': round(volatility, 2),
                    'sharpe_ratio': round(sharpe_ratio, 4),
                    'karma_ratio': round(karma_ratio, 4),
                    }
        except Exception as e:
            print("generate_profit_statistics错误原因:{},行数:{}".format(e, e.__traceback__.tb_lineno))

    def generate_profit_statistics_year_profit(self, symbol):
        """
            year_profit #年收益
            year_profit_rate #年化收益率
        """
        total_profit = self.profit_data[symbol]['total_profit']
        # print(self.profit_data[symbol]['timestamps'])
        if self.profit_data[symbol]['timestamps']['start_time'] and self.profit_data[symbol]['timestamps']['now_time']:
            if self.profit_data[symbol]['timestamps']['now_time'] - self.profit_data[symbol]['timestamps']['start_time'] != 0:
                year_profit = round(total_profit / ((self.profit_data[symbol]['timestamps']['now_time'] - self.profit_data[symbol]['timestamps']['start_time']) / self.sqrt_base_hour / 24) * 365, 6)
                year_profit_rate = round((year_profit / self.principal[symbol]) * 100, 2)
                return {'year_profit': year_profit, 'year_profit_rate': year_profit_rate}
            else:
                """当收益时间差为0时，年收益和年收益率为'∞'"""
                year_profit = round(total_profit, 6)
                if year_profit > 0:
                    year_profit_rate = 0
                else:
                    year_profit_rate = 0
                return {'year_profit': year_profit, 'year_profit_rate': year_profit_rate}
        else:
            """当时间不存在时正无穷"""
            year_profit = round(total_profit, 6)
            if year_profit > 0:
                year_profit_rate = 0
            else:
                year_profit_rate = 0
            return {'year_profit': year_profit, 'year_profit_rate': year_profit_rate}

    def generate_profit_list_val_list(self, symbol):
        profit_list = self.profit_data[symbol]['profit_list']
        rtn_list = list()
        if len(profit_list) > 0:
            for i in range(len(profit_list) - 1):
                rtn_list.append((profit_list[i]+self.principal[symbol]) / self.principal[symbol])
        return rtn_list


    def generate_profit_statistics_volatility(self, symbol, sqrt_base):
        """利润波动率:指投资回报率在过去一段时间内所表现出的波动率"""
        val_list = self.generate_profit_list_val_list(symbol)
        if len(val_list) > 0:
            rtn_list = list()
            for i in range(len(val_list) - 1):
                rtn_list.append(val_list[i + 1] - val_list[i])
            return round(np.std(val_list, ddof=1) * np.sqrt(sqrt_base),6)
        else:
            return 0

    def generate_profit_statistics_sharpe_ratio(self, symbol, sqrt_base, year_profit_rate):
        """夏普比例：(年华收益率 - 无风险收益率) / 年化波动率"""
        val_list = self.generate_profit_list_val_list(symbol)
        if len(val_list) > 1:
            rtn_list = list()
            for i in range(len(val_list) - 1):
                rtn_list.append(val_list[i + 1] - val_list[i])
            return round(sum(rtn_list) / len(rtn_list) / np.std(rtn_list, ddof=1) * np.sqrt(sqrt_base),6)
        else:
            return 0
        # if len(self.profit_data[symbol]['profit_list']) == 0:
        #     return None
        # else:
        #     return round(sum(self.profit_data[symbol]['profit_list']) / len(self.profit_data[symbol]['profit_list']) / np.std(self.profit_data[symbol]['profit_list'], ddof=1) * np.sqrt(sqrt_base), 6)

    def generate_profit_statistics_karma_ratio(self, symbol):
        """卡玛比率：超额收益/最大回测"""
        if self.profit_data[symbol]['max_back'] > 0:
            if self.profit_data[symbol]['total_profit'] != 0:
                return round(self.profit_data[symbol]['total_profit'] / self.profit_data[symbol]['max_back'], 6)
            else:
                return 0
        else:
            return 0

    def generate_profit_statistics_max_back(self, symbol):
        profit_list = self.profit_data[symbol]['profit_list']
        max_back_rate = 0
        max_val = 0
        max_back = 0
        if len(profit_list) > 0:
            for val in profit_list:
                max_val = max(max_val, val)
                # max_back = min(max_back, val - max_val)
                max_back_rate = max(max_back_rate, round(abs(val - max_val) / (max_val + self.principal[symbol]) * 100, 6))
        return max_back_rate

    def generate_profit_total_times(self,symbol):
        count_profit = self.profit_data[symbol]['count_profit']
        total_times = count_profit['buy_time'] + count_profit['sell_time']
        win_times = count_profit['buy_+_time'] + count_profit['sell_+_time']
        if total_times != 0:
            wrate = round((win_times / total_times) * 100, 4)
            return {'total_times': total_times, 'wrate': wrate}
        else:
            return {'total_times': 0, 'wrate': 0}


if __name__ == '__main__':
    profile = Profit(['eth'])
    order1 = {'o_type': 1, 'price': 2607.1, 'side': 'sell', 'size': 0.1, 'timestamp': int(1), 'symbol': 'eth'}
    order2 = {'o_type': 1, 'price': 2612.3, 'side': 'sell', 'size': 0.1, 'timestamp': int(1), 'symbol': 'eth'}
    order3 = {'o_type': 1, 'price': 2601.2, 'side': 'sell', 'size': 0.1, 'timestamp': int(1), 'symbol': 'eth'}
    order4 = {'o_type': 1, 'price': 2607.1, 'side': 'buy', 'size': 0.3, 'timestamp': int(1), 'symbol': 'eth'}
    order5 = {'o_type': 1, 'price': 2602.4, 'side': 'buy', 'size': 0.1, 'timestamp': int(1), 'symbol': 'eth'}
    order6 = {'o_type': 1, 'price': 2599.3, 'side': 'sell', 'size': 0.1, 'timestamp': int(1), 'symbol': 'eth'}
    order_list = [order1, order2, order3, order4,order5,order6]
    orders = [{'o_type': 1, 'price': 722.7433859999999, 'side': 'buy', 'size': 1.0, 'timestamp': 1609548000, 'symbol': 'eth'}, {'o_type': 1, 'price': 720.7321799999999, 'side': 'buy', 'size': 1.0, 'timestamp': 1609548300, 'symbol': 'eth'}, {'o_type': 1, 'price': 716.2294799999999, 'side': 'buy', 'size': 1.0, 'timestamp': 1609549800, 'symbol': 'eth'}, {'o_type': 1, 'price': 720.5474120000001, 'side': 'sell', 'size': 3.0, 'timestamp': 1609550100, 'symbol': 'eth'}, {'o_type': 1, 'price': 1194.5462979999998, 'side': 'buy', 'size': 1.0, 'timestamp': 1610046000, 'symbol': 'eth'}, {'o_type': 1, 'price': 1207.0353440000001, 'side': 'sell', 'size': 1.0, 'timestamp': 1610046900, 'symbol': 'eth'}, {'o_type': 1, 'price': 1313.441462, 'side': 'sell', 'size': 1.0, 'timestamp': 1611813600, 'symbol': 'eth'}, {'o_type': 1, 'price': 1309.305112, 'side': 'buy', 'size': 1.0, 'timestamp': 1611815700, 'symbol': 'eth'}, {'o_type': 1, 'price': 1970.5416159999997, 'side': 'buy', 'size': 1.0, 'timestamp': 1613852100, 'symbol': 'eth'}, {'o_type': 1, 'price': 1983.269324, 'side': 'sell', 'size': 1.0, 'timestamp': 1613852700, 'symbol': 'eth'}, {'o_type': 1, 'price': 1567.369014, 'side': 'sell', 'size': 1.0, 'timestamp': 1615046700, 'symbol': 'eth'}, {'o_type': 1, 'price': 1580.2976099999998, 'side': 'buy', 'size': 1.0, 'timestamp': 1615050000, 'symbol': 'eth'}, {'o_type': 1, 'price': 2087.5317680000003, 'side': 'buy', 'size': 1.0, 'timestamp': 1617770100, 'symbol': 'eth'}, {'o_type': 1, 'price': 2091.204524, 'side': 'sell', 'size': 1.0, 'timestamp': 1617770700, 'symbol': 'eth'}, {'o_type': 1, 'price': 1929.567046, 'side': 'buy', 'size': 1.0, 'timestamp': 1624606200, 'symbol': 'eth'}, {'o_type': 1, 'price': 1937.9665220000002, 'side': 'sell', 'size': 1.0, 'timestamp': 1624607400, 'symbol': 'eth'}, {'o_type': 1, 'price': 2075.5938960000003, 'side': 'sell', 'size': 1.0, 'timestamp': 1624892400, 'symbol': 'eth'}, {'o_type': 1, 'price': 2116.119566, 'side': 'sell', 'size': 1.0, 'timestamp': 1624895700, 'symbol': 'eth'}, {'o_type': 1, 'price': 2109.775106, 'side': 'buy', 'size': 2.0, 'timestamp': 1624896900, 'symbol': 'eth'}, {'o_type': 1, 'price': 1940.055268, 'side': 'sell', 'size': 1.0, 'timestamp': 1626568500, 'symbol': 'eth'}, {'o_type': 1, 'price': 1969.0578560000001, 'side': 'sell', 'size': 1.0, 'timestamp': 1626569700, 'symbol': 'eth'}, {'o_type': 1, 'price': 1976.615258, 'side': 'buy', 'size': 2.0, 'timestamp': 1626572100, 'symbol': 'eth'}, {'o_type': 1, 'price': 2427.5326059999998, 'side': 'sell', 'size': 1.0, 'timestamp': 1627327200, 'symbol': 'eth'}, {'o_type': 1, 'price': 2400.219268, 'side': 'buy', 'size': 1.0, 'timestamp': 1627328100, 'symbol': 'eth'}, {'o_type': 1, 'price': 2869.37734, 'side': 'sell', 'size': 1.0, 'timestamp': 1628265600, 'symbol': 'eth'}, {'o_type': 1, 'price': 2867.959744, 'side': 'buy', 'size': 1.0, 'timestamp': 1628268000, 'symbol': 'eth'}, {'o_type': 1, 'price': 4018.0277360000005, 'side': 'sell', 'size': 1.0, 'timestamp': 1630679400, 'symbol': 'eth'}, {'o_type': 1, 'price': 3974.643356, 'side': 'buy', 'size': 1.0, 'timestamp': 1630679700, 'symbol': 'eth'}, {'o_type': 1, 'price': 3415.84828, 'side': 'buy', 'size': 1.0, 'timestamp': 1631028000, 'symbol': 'eth'}, {'o_type': 1, 'price': 3372.552318, 'side': 'buy', 'size': 1.0, 'timestamp': 1631028300, 'symbol': 'eth'}, {'o_type': 1, 'price': 3380.960206, 'side': 'sell', 'size': 2.0, 'timestamp': 1631028600, 'symbol': 'eth'}, {'o_type': 1, 'price': 3322.7624619999997, 'side': 'buy', 'size': 1.0, 'timestamp': 1632052800, 'symbol': 'eth'}, {'o_type': 1, 'price': 3342.0535640000003, 'side': 'sell', 'size': 1.0, 'timestamp': 1632054000, 'symbol': 'eth'}, {'o_type': 1, 'price': 3539.934764, 'side': 'sell', 'size': 1.0, 'timestamp': 1633525800, 'symbol': 'eth'}, {'o_type': 1, 'price': 3570.7211479999996, 'side': 'buy', 'size': 1.0, 'timestamp': 1633529100, 'symbol': 'eth'}, {'o_type': 1, 'price': 4755.954714, 'side': 'sell', 'size': 1.0, 'timestamp': 1636376400, 'symbol': 'eth'}, {'o_type': 1, 'price': 4739.672098, 'side': 'buy', 'size': 1.0, 'timestamp': 1636377300, 'symbol': 'eth'}]
    profile.profit_every_order(orders)
    pprint(profile.position)
    pprint(profile.profit_data)
    print(profile.generate_profit_statistics('eth'))
