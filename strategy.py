

class BaseStrategy(object):
    def __init__(self, engine):
        self.engine = engine

    def set_engin(self, engine):
        self.engine = engine

    def notify_order(self, order):
        """
        当订单发生变化时，接收订单
        """
        pass

    def notify_trade(self, trade):
        """
        在交易发生变化时接收交易
        """
        pass

    def notify_data(self, data, *args, **kwargs):
        """接收来自数据的通知"""
        pass

    def notify_position(self, data):
        """在仓位利润发生变化室,接受信息数据"""
        pass

class AiStrategy(BaseStrategy):

    a
    def __int__(self):
        pass

    def notify_order(self, order):
        pass

    def notify_trade(self, trade):
        pass

    def notify_position(self, data):
        pass

    def notify_data(self, data, *args, **kwargs):
        """
                @param symbol:              'btcusdt'
                @param price:                bid_price or ask_price
                @param side:                'buy' or 'sell'
                @param size:                数量
                @param order_type:
                                            gtc: GoodTillCancelled
                                            ioc: ImmediateOrCancelled，立即成交或者取消，只吃单不挂单
                                            poc: PendingOrCancelled，被动委托，只挂单不吃单
                                            fok: FillOrKill，全部成交或者全部取消
                @param o_type:              o_type = 0 则不进行反向跟单 1 则是跟反向taker订单 2则是跟反向maker订单 3表示撤单后跟同向的taker
                @param create_timestamp:    当前时间戳 / 创建订单时间戳
                @param auto_cancel: False,  是否自动撤单
                @param auto_interval: 0,    撤单间隔, 当auto_cancel为True时才生效
                :return:
                    {'id':1, 'symbol': symbol,  'price': price, 'side': side,  'size': size, 'order_type': 'taker','o_type': 1,
                     'create_timestamp': int(create_timestamp), 'update_timestamp': int(1368544970), 'filled_size':0,
                     'auto_interval': False, 'auto_interval':0, 'status': 'open'}
                update_timestamp:           更新时间戳 / 也是平仓时间戳
                filled_size: 成交数量
                auto_interval：自动撤单
                auto_interval：自动撤单间隔
                status： 订单状态 open 开仓, closed 完全成交, cancelled 订单撤销,
        """

        self.engine.orderCl.place_order('btcusdt', 1364.2, 'buy', 0.01, 'poc', 0, 1357685270)

