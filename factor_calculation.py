import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

def calc_wap12(df):
    var1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    var2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_wap34(df):
    var1 = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']
    var2 = df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_swap1(df):
    return df['wap1'] - df['wap3']

def calc_swap12(df):
    return df['wap12'] - df['wap34']

def calc_tswap1(df):
    return -df['swap1'].diff()

def calc_tswap12(df):
    return -df['swap12'].diff()

def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    return (ask - bid) / df['midprice']

def calc_tt1(df):
    p1 = df['ask_price1'] * df['ask_size1'] + df['bid_price1'] * df['bid_size1']
    p2 = df['ask_price2'] * df['ask_size2'] + df['bid_price2'] * df['bid_size2']
    return p2 - p1

def calc_price_impact(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    return (df['ask_price1'] - ask)/df['ask_price1'], (df['bid_price1'] - bid)/df['bid_price1']

# Calculate order book slope
def calc_slope(df):
    v0 = (df['bid_size1']+df['ask_size1'])/2
    p0 = (df['bid_price1']+df['ask_price1'])/2
    slope_bid = ((df['bid_size1']/v0)-1)/abs((df['bid_price1']/p0)-1)+(
                (df['bid_size2']/df['bid_size1'])-1)/abs((df['bid_price2']/df['bid_price1'])-1)
    slope_ask = ((df['ask_size1']/v0)-1)/abs((df['ask_price1']/p0)-1)+(
                (df['ask_size2']/df['ask_size1'])-1)/abs((df['ask_price2']/df['ask_price1'])-1)
    return (slope_bid+slope_ask)/2, abs(slope_bid-slope_ask)

# Calculate order book dispersion
def calc_dispersion(df):
    bspread = df['bid_price1'] - df['bid_price2']
    aspread = df['ask_price2'] - df['ask_price1']
    bmid = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price1']
    bmid2 = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price2']
    amid = df['ask_price1'] - (df['bid_price1'] + df['ask_price1'])/2
    amid2 = df['ask_price2'] - (df['bid_price1'] + df['ask_price1'])/2
    bdisp = (df['bid_size1']*bmid + df['bid_size2']*bspread)/(df['bid_size1']+df['bid_size2'])
    bdisp2 = (df['bid_size1']*bmid + df['bid_size2']*bmid2)/(df['bid_size1']+df['bid_size2'])
    adisp = (df['ask_size1']*amid + df['ask_size2']*aspread)/(df['ask_size1']+df['ask_size2'])
    adisp2 = (df['ask_size1']*amid + df['ask_size2']*amid2)/(df['ask_size1']+df['ask_size2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp)/2, (bdisp2 + adisp2)/2


# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
               'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth

#  order flow imbalance
def calc_ofi(df):
    a = df['bid_size1']*np.where(df['bid_price1'].diff()>=0,1,0)
    b = df['bid_size1'].shift()*np.where(df['bid_price1'].diff()<=0,1,0)
    c = df['ask_size1']*np.where(df['ask_price1'].diff()<=0,1,0)
    d = df['ask_size1'].shift()*np.where(df['ask_price1'].diff()>=0,1,0)
    return a - b - c + d

#%%
# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()
#%%
# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    return (np.sum(series**4)*series.shape[0]/3)

def reciprocal_transformation(series):
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)

# Calculate the realized absolute variation
def realized_absvar(series):
    return np.sqrt(np.pi/(2*series.count()))*np.sum(np.abs(series))

# Calculate the realized skew
def realized_skew(series):
    return np.sqrt(series.count())*np.sum(series**3)/(realized_volatility(series)**3)

# Calculate the realized kurtosis
def realized_kurtosis(series):
    return series.count()*np.sum(series**4)/(realized_volatility(series)**4)
#%%  计算订单簿因子，数据间隔为每秒
def book_preprocessor(data):

    df = data

    rolling = 60

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)

    df['wap12'] = calc_wap12(df)
    df['wap34'] = calc_wap34(df)

    df['swap1'] = calc_swap1(df)
    df['swap12'] = calc_swap12(df)

    df['tswap1'] = calc_tswap1(df)
    df['tswap12'] = calc_tswap12(df)

    df['wss12'] = calc_wss12(df)
    df['tt1'] = calc_tt1(df)

    df['price_impact1'], df['price_impact2'] = calc_price_impact(df)

    df['slope1'], df['slope2'] = calc_slope(df)

    df['bspread'] = df['bid_price1'] - df['bid_price2']
    df['aspread'] = df['ask_price2'] - df['ask_price1']
    df['bmid'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price1']
    df['bmid2'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price2']
    df['amid'] = df['ask_price1'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['amid2'] = df['ask_price2'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['bdisp'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bspread']) / (df['bid_size1'] + df['bid_size2'])
    df['bdisp2'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bmid2']) / (df['bid_size1'] + df['bid_size2'])
    df['adisp'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['aspread']) / (df['ask_size1'] + df['ask_size2'])
    df['adisp2'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['amid2']) / (df['ask_size1'] + df['ask_size2'])

    df['depth'] = calc_depth(df)

    df['ofi'] = calc_ofi(df)

    # wap1 genetic functions
    # realized volatility
    df['wap1_volatility1'] = df['wap1'].rollong(rolling).apply(realized_volatility)
    df['wap1_volatility20'] = df['wap1'].rolling(rolling*20).apply(realized_volatility)
    df['wap1_volatility100'] = df['wap1'].rolling(rolling*100).apply(realized_volatility)
    # realized quarticity
    df['wap1_quarticity1'] = df['wap1'].rollong(rolling).apply(realized_quarticity)
    df['wap1_quarticity20'] = df['wap1'].rolling(rolling * 20).apply(realized_quarticity)
    df['wap1_quarticity100'] = df['wap1'].rolling(rolling * 100).apply(realized_quarticity)
    # reciprocal transformation
    df['wap1_reciprocal1'] = df['wap1'].rollong(rolling).apply(reciprocal_transformation)
    df['wap1_reciprocal20'] = df['wap1'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['wap1_reciprocal100'] = df['wap1'].rolling(rolling * 100).apply(reciprocal_transformation)
    # square root translation
    df['wap1_square_root1'] = df['wap1'].rollong(rolling).apply(square_root_translation)
    df['wap1_square_root20'] = df['wap1'].rolling(rolling * 20).apply(square_root_translation)
    df['wap1_square_root100'] = df['wap1'].rolling(rolling * 100).apply(square_root_translation)
    # realized absvar
    df['wap1_absvar1'] = df['wap1'].rollong(rolling).apply(realized_absvar)
    df['wap1_absvar20'] = df['wap1'].rollong(rolling * 20).apply(realized_absvar)
    df['wap1_absvar100'] = df['wap1'].rollong(rolling * 100).apply(realized_absvar)
    # realized skew
    df['wap1_skew1'] = df['wap1'].rollong(rolling).apply(realized_skew)
    df['wap1_skew20'] = df['wap1'].rollong(rolling * 20).apply(realized_skew)
    df['wap1_skew100'] = df['wap1'].rollong(rolling * 100).apply(realized_skew)
    # realized kurtosis
    df['wap1_kurtosis1'] = df['wap1'].rollong(rolling).apply(realized_kurtosis)
    df['wap1_kurtosis20'] = df['wap1'].rollong(rolling * 20).apply(realized_kurtosis)
    df['wap1_kurtosis100'] = df['wap1'].rollong(rolling * 100).apply(realized_kurtosis)

    # wap2 genetic functions
    # realized volatility
    df['wap2_volatility1'] = df['wap2'].rollong(rolling).apply(realized_volatility)
    df['wap2_volatility20'] = df['wap2'].rolling(rolling * 20).apply(realized_volatility)
    df['wap2_volatility100'] = df['wap2'].rolling(rolling * 100).apply(realized_volatility)
    # realized quarticity
    df['wap2_quarticity1'] = df['wap2'].rollong(rolling).apply(realized_quarticity)
    df['wap2_quarticity20'] = df['wap2'].rolling(rolling * 20).apply(realized_quarticity)
    df['wap2_quarticity100'] = df['wap2'].rolling(rolling * 100).apply(realized_quarticity)
    # reciprocal transformation
    df['wap2_reciprocal1'] = df['wap2'].rollong(rolling).apply(reciprocal_transformation)
    df['wap2_reciprocal20'] = df['wap2'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['wap2_reciprocal100'] = df['wap2'].rolling(rolling * 100).apply(reciprocal_transformation)
    # square root translation
    df['wap2_square_root1'] = df['wap2'].rollong(rolling).apply(square_root_translation)
    df['wap2_square_root20'] = df['wap2'].rolling(rolling * 20).apply(square_root_translation)
    df['wap2_square_root100'] = df['wap2'].rolling(rolling * 100).apply(square_root_translation)
    # realized absvar
    df['wap2_absvar1'] = df['wap2'].rollong(rolling).apply(realized_absvar)
    df['wap2_absvar20'] = df['wap2'].rollong(rolling * 20).apply(realized_absvar)
    df['wap2_absvar100'] = df['wap2'].rollong(rolling * 100).apply(realized_absvar)
    # realized skew
    df['wap2_skew1'] = df['wap2'].rollong(rolling).apply(realized_skew)
    df['wap2_skew20'] = df['wap2'].rollong(rolling * 20).apply(realized_skew)
    df['wap2_skew100'] = df['wap2'].rollong(rolling * 100).apply(realized_skew)
    # realized kurtosis
    df['wap2_kurtosis1'] = df['wap2'].rollong(rolling).apply(realized_kurtosis)
    df['wap2_kurtosis20'] = df['wap2'].rollong(rolling * 20).apply(realized_kurtosis)
    df['wap2_kurtosis100'] = df['wap2'].rollong(rolling * 100).apply(realized_kurtosis)

    # wap3 genetic functions
    # realized volatility
    df['wap3_volatility1'] = df['wap3'].rollong(rolling).apply(realized_volatility)
    df['wap3_volatility20'] = df['wap3'].rolling(rolling * 20).apply(realized_volatility)
    df['wap3_volatility100'] = df['wap3'].rolling(rolling * 100).apply(realized_volatility)
    # realized quarticity
    df['wap3_quarticity1'] = df['wap3'].rollong(rolling).apply(realized_quarticity)
    df['wap3_quarticity20'] = df['wap3'].rolling(rolling * 20).apply(realized_quarticity)
    df['wap3_quarticity100'] = df['wap3'].rolling(rolling * 100).apply(realized_quarticity)
    # reciprocal transformation
    df['wap3_reciprocal1'] = df['wap3'].rollong(rolling).apply(reciprocal_transformation)
    df['wap3_reciprocal20'] = df['wap3'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['wap3_reciprocal100'] = df['wap3'].rolling(rolling * 100).apply(reciprocal_transformation)
    # square root translation
    df['wap3_square_root1'] = df['wap3'].rollong(rolling).apply(square_root_translation)
    df['wap3_square_root20'] = df['wap3'].rolling(rolling * 20).apply(square_root_translation)
    df['wap3_square_root100'] = df['wap3'].rolling(rolling * 100).apply(square_root_translation)
    # realized absvar
    df['wap3_absvar1'] = df['wap3'].rollong(rolling).apply(realized_absvar)
    df['wap3_absvar20'] = df['wap3'].rollong(rolling * 20).apply(realized_absvar)
    df['wap3_absvar100'] = df['wap3'].rollong(rolling * 100).apply(realized_absvar)
    # realized skew
    df['wap3_skew1'] = df['wap3'].rollong(rolling).apply(realized_skew)
    df['wap3_skew20'] = df['wap3'].rollong(rolling * 20).apply(realized_skew)
    df['wap3_skew100'] = df['wap3'].rollong(rolling * 100).apply(realized_skew)
    # realized kurtosis
    df['wap3_kurtosis1'] = df['wap3'].rollong(rolling).apply(realized_kurtosis)
    df['wap3_kurtosis20'] = df['wap3'].rollong(rolling * 20).apply(realized_kurtosis)
    df['wap3_kurtosis100'] = df['wap3'].rollong(rolling * 100).apply(realized_kurtosis)

    # wap4 genetic functions
    # realized volatility
    df['wap4_volatility1'] = df['wap4'].rollong(rolling).apply(realized_volatility)
    df['wap4_volatility20'] = df['wap4'].rolling(rolling * 20).apply(realized_volatility)
    df['wap4_volatility100'] = df['wap4'].rolling(rolling * 100).apply(realized_volatility)
    # realized quarticity
    df['wap4_quarticity1'] = df['wap4'].rollong(rolling).apply(realized_quarticity)
    df['wap4_quarticity20'] = df['wap4'].rolling(rolling * 20).apply(realized_quarticity)
    df['wap4_quarticity100'] = df['wap4'].rolling(rolling * 100).apply(realized_quarticity)
    # reciprocal transformation
    df['wap4_reciprocal1'] = df['wap4'].rollong(rolling).apply(reciprocal_transformation)
    df['wap4_reciprocal20'] = df['wap4'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['wap4_reciprocal100'] = df['wap4'].rolling(rolling * 100).apply(reciprocal_transformation)
    # square root translation
    df['wap4_square_root1'] = df['wap4'].rollong(rolling).apply(square_root_translation)
    df['wap4_square_root20'] = df['wap4'].rolling(rolling * 20).apply(square_root_translation)
    df['wap4_square_root100'] = df['wap4'].rolling(rolling * 100).apply(square_root_translation)
    # realized absvar
    df['wap4_absvar1'] = df['wap4'].rollong(rolling).apply(realized_absvar)
    df['wap4_absvar20'] = df['wap4'].rollong(rolling * 20).apply(realized_absvar)
    df['wap4_absvar100'] = df['wap4'].rollong(rolling * 100).apply(realized_absvar)
    # realized skew
    df['wap4_skew1'] = df['wap4'].rollong(rolling).apply(realized_skew)
    df['wap4_skew20'] = df['wap4'].rollong(rolling * 20).apply(realized_skew)
    df['wap4_skew100'] = df['wap4'].rollong(rolling * 100).apply(realized_skew)
    # realized kurtosis
    df['wap4_kurtosis1'] = df['wap4'].rollong(rolling).apply(realized_kurtosis)
    df['wap4_kurtosis20'] = df['wap4'].rollong(rolling * 20).apply(realized_kurtosis)
    df['wap4_kurtosis100'] = df['wap4'].rollong(rolling * 100).apply(realized_kurtosis)

    df['wap1_shift2'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['wap1_shift5'] = df['wap1'].shift(1) - df['wap1'].shift(5)
    df['wap1_shift10'] = df['wap1'].shift(1) - df['wap1'].shift(10)

    df['wap2_shift15'] = df['wap2'].shift(1) - df['wap2'].shift(2)
    df['wap2_shift20'] = df['wap2'].shift(1) - df['wap2'].shift(5)
    df['wap2_shift30'] = df['wap2'].shift(1) - df['wap2'].shift(10)


    df['wap3_shift2'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['wap3_shift5'] = df['wap3'].shift(1) - df['wap3'].shift(5)
    df['wap3_shift10'] = df['wap3'].shift(1) - df['wap3'].shift(10)


    df['wap4_shift2'] = df['wap4'].shift(1) - df['wap4'].shift(2)
    df['wap4_shift5'] = df['wap4'].shift(1) - df['wap4'].shift(5)
    df['wap4_shift10'] = df['wap4'].shift(1) - df['wap4'].shift(10)


    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df.ask_price1==df.ask_price1.shift(1),df.ask_size1-df.ask_size1.shift(1),0)
    df['vtA'] = np.where(df.ask_price1>df.ask_price1.shift(1),df.ask_size1,df.pre_vtA)
    df['pre_vtB'] = np.where(df.bid_price1==df.bid_price1.shift(1),df.bid_size1-df.bid_size1.shift(1),0)
    df['vtB'] = np.where(df.bid_price1>df.bid_price1.shift(1),df.bid_size1,df.pre_vtB)

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

    # Calculate log returns


    df['roliing_mid_price1_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['rolling_mid_price1_std'] = df['mid_price1'].rolling(rolling).std()

    df['rolling_HR1_mean'] = df['HR1'].rolling(rolling).mean()

    df['rolling_bid_ask_size1_minus_mean1'] = df['bid_ask_size1_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size2_minus_mean1'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size3_minus_mean1'] = df['bid_ask_size3_minus'].rolling(rolling).mean()


    df['rolling_bid_size1_shift_mean1'] = df['bid_size1_shift'].rolling(rolling).mean()
    df['rolling_bid_size1_shift_mean20'] = df['bid_size1_shift'].rolling(20*rolling).mean()
    df['rolling_bid_size1_shift_mean100'] = df['bid_size1_shift'].rolling(100*rolling).mean()
    df['rolling_ask_size1_shift_mean1'] = df['ask_size1_shift'].rolling(rolling).mean()
    df['rolling_ask_size1_shift_mean2'] = df['ask_size1_shift'].rolling(20*rolling).mean()
    df['rolling_ask_size1_shift_mean100'] = df['ask_size1_shift'].rolling(100*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean1'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_spread_mean20'] = df['bid_ask_size1_spread'].rolling(20*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean100'] = df['bid_ask_size1_spread'].rolling(100*rolling).mean()


    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return2'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(2)) * 100
    df['log_return3'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(2)) * 100
    df['log_return4'] = np.log(df['wap4'].shift(1)/df['wap4'].shift(2))*100


    df['log_return_wap1_shift5'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(5))*100
    df['log_return_wap2_shift5'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(5)) * 100
    df['log_return_wap3_shift5'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(5)) * 100
    df['log_return_wap4_shift5'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(5)) * 100

    df['log_return_wap1_shift15'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(15))*100
    df['log_return_wap2_shift15'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(15)) * 100
    df['log_return_wap3_shift15'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(15)) * 100
    df['log_return_wap4_shift15'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(15)) * 100


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

    print(df.columns)

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

    # last price genetic functions
    # log_return
    df['log_return'] = df.grouby('datetime')['last_price'].apply(log_return)
    # realized volatility
    df['log_return_volatility1'] = df['log_return'].rollong(rolling).apply(realized_volatility)
    df['log_return_volatility20'] = df['log_return'].rolling(rolling * 20).apply(realized_volatility)
    df['log_return_volatility100'] = df['log_return'].rolling(rolling * 100).apply(realized_volatility)
    # realized quarticity
    df['log_return_quarticity1'] = df['log_return'].rollong(rolling).apply(realized_quarticity)
    df['log_return_quarticity20'] = df['log_return'].rolling(rolling * 20).apply(realized_quarticity)
    df['log_return_quarticity100'] = df['log_return'].rolling(rolling * 100).apply(realized_quarticity)
    # reciprocal transformation
    df['wap1_reciprocal1'] = df['log_return'].rollong(rolling).apply(reciprocal_transformation)
    df['wap1_reciprocal20'] = df['log_return'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['wap1_reciprocal100'] = df['log_return'].rolling(rolling * 100).apply(reciprocal_transformation)
    # square root translation
    df['log_return_square_root1'] = df['log_return'].rollong(rolling).apply(square_root_translation)
    df['log_return_square_root20'] = df['log_return'].rolling(rolling * 20).apply(square_root_translation)
    df['log_return_square_root100'] = df['log_return'].rolling(rolling * 100).apply(square_root_translation)
    # realized absvar
    df['log_return_absvar1'] = df['log_return'].rollong(rolling).apply(realized_absvar)
    df['log_return_absvar20'] = df['log_return'].rollong(rolling * 20).apply(realized_absvar)
    df['log_return_absvar100'] = df['log_return'].rollong(rolling * 100).apply(realized_absvar)
    # realized skew
    df['log_return_skew1'] = df['log_return'].rollong(rolling).apply(realized_skew)
    df['log_return_skew20'] = df['log_return'].rollong(rolling * 20).apply(realized_skew)
    df['log_return_skew100'] = df['log_return'].rollong(rolling * 100).apply(realized_skew)

    df['mid_price'] = np.where(df.size > 0, (df.amount - df.amount.shift(1)) / df.size, df.last_price)
    df['rolling_mid_price_mean'] = df['mid_price'].rolling(rolling).mean()
    df['rolling_mid_price_std'] = df['mid_price'].rolling(rolling).std()

    df['last_price_shift1'] = df['last_price'].shift(1) - df['last_price'].shift(1)
    df['last_price_shift2'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift5'] = df['last_price'].shift(1) - df['last_price'].shift(5)
    df['last_price_shift10'] = df['last_price'].shift(1) - df['last_price'].shift(10)


    df['log_return_last_price_shift1'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(1)) * 100
    df['log_return_last_price_shift2'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(2)) * 100
    df['log_return_last_price_shift5'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(5)) * 100
    df['log_return_last_price_shift10'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(10)) * 100


    # size genetic functions
    # realized skew
    df['size_skew1'] = df['size'].rolling(rolling).apply(realized_skew)
    df['size_skew20'] = df['size'].rolling(rolling * 20).apply(realized_skew)
    df['size_skew100'] = df['size'].rolling(rolling * 100).apply(realized_skew)
    # realized kurtosis
    df['size_kurtosis1'] = df['size'].rolling(rolling).apply(realized_kurtosis)
    df['size_kurtosis20'] = df['size'].rolling(rolling * 20).apply(realized_kurtosis)
    df['size_kurtosis100'] = df['size'].rolling(rolling * 100).apply(realized_kurtosis)
    # realized absvar
    df['size_absvar1'] = df['size'].rolling(rolling).apply(realized_absvar)
    df['size_absvar20'] = df['size'].rolling(rolling * 20).apply(realized_absvar)
    df['size_absvar100'] = df['size'].rolling(rolling * 100).apply(realized_absvar)
    # realized quarticity
    df['size_quarticity1'] = df['size'].rolling(rolling).apply(realized_quarticity)
    df['size_quarticity20'] = df['size'].rolling(rolling * 20).apply(realized_quarticity)
    df['size_quarticity100'] = df['size'].rolling(rolling * 100).apply(realized_quarticity)
    # square root translation
    df['size_square_root1'] = df['size'].rolling(rolling).apply(square_root_translation)
    df['size_square_root20'] = df['size'].rolling(rolling * 20).apply(square_root_translation)
    df['size_square_root100'] = df['size'].rolling(rolling * 100).apply(square_root_translation)
    # reciprocal_transformation
    df['size_reciprocal1'] = df['size'].rolling(rolling).apply(reciprocal_transformation)
    df['size_reciprocal20'] = df['size'].rolling(rolling * 20).apply(reciprocal_transformation)
    df['size_reciprocal100'] = df['size'].rolling(rolling * 100).apply(reciprocal_transformation)

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
    df['size_percentile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['size_percentile'] = df['size_percentile_75'] - df['size_percentile_25']


    df['price_percentile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['price_percentile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['price_percentile'] = df['price_percentile_75'] - df['price_percentile_25']



    df['rolling_mean_amount'] = df['amount'].rolling(rolling).mean()
    df['rolling_quantile_25_amount'] = df['amount'].rolling(rolling).quantile(.25)
    df['rolling_quantile_50_amount'] = df['amount'].rolling(rolling).quantile(.50)
    df['rolling_quantile_75_amount'] = df['amount'].rolling(rolling).quantile(.75)


    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()

    print(df.columns)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df
#%% 计算输出因子

book_data = book_preprocessor()
trade_data = trade_preprocessor()
#%%
book_data['datetime'] = pd.to_datetime(book_data['datetime'])
trade_data['datetime'] = pd.to_datetime(trade_data['datetime'])
#%% 计算这一分钟的vwap为target,且为挂单价格
def get_vwap(data):
    v = data['szie']
    p = data['last_price']
    data['last_price_vwap'] = np.sum(p*v) / np.sum(v)
    return data

time_group = trade_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).apply(get_vwap)
# 通过trade产出的因子将数据进项聚合为1分钟的数据
time_group_trade = time_group.groupby(pd.Grouper(freq='1min')).agg(np.mean)
time_group_trade = time_group_trade.dropna(axis=0,how='all')
time_group_trade = time_group_trade.reset_index()    #time_group为最终传入模型的因子
#%% 通过book产出的因子聚合为1分钟的数据
time_group_book = book_data.set_index('datetime').groupby(pd.Grouper(freq='1min')).agg(np.mean)
time_group_book = time_group_book.dropna(axis=0,how='all')
time_group_book = time_group_book.reset_index()
#%%
time_group_book = pd.read_csv('time_group_book.csv')
time_group_trade = pd.read_csv('time_group_trade.csv')
final_data = pd.merge(time_group_trade, time_group_book, on='datetime', how='left')