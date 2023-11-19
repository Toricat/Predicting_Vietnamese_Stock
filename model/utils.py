import pandas as pd
import numpy as np

import vnquant.data as dt
from vnquant import plot as pl
from vnquant import utils

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller

class SelectedStock:
    """SelectedStock
    
    """
    def __init__(self, stock, selected_stock_start, selected_stock_end,data_selected=None):
        self.stock = stock
        self.start = selected_stock_start
        self.end = selected_stock_end
        self.data = data_selected if data_selected is not None else self.stock_data()

    def stock_data(self):
        loader = dt.DataLoader(symbols=[self.stock], 
                               start=self.start,
                               end=self.end,
                               minimal=True, 
                               data_source="VND")
        self.data = loader.download()
        full_date_rng =  pd.date_range(start=self.data.index[0], end= self.data.index[-1], freq='D')
        self.data = self.data.reindex(full_date_rng)
        return self.data
    
    def stock_chart(self):
       
        pl.vnquant_candle_stick(data=self.data,
                                title=f'Your Stock ({self.stock}) from {self.start} to {self.end}',
                                xlab='Date', ylab='Values',
                                show_advanced=['volume', 'macd', 'rsi'])
    def stock_infor(self,row = None):
        print(self.data.head(row))
        is_ohlc = utils._isOHLC(self.data)
        is_ohlcv = utils._isOHLCV(self.data)
        print(f"stock {self.stock} is OHLC: '{is_ohlc}' and OHLCV: '{is_ohlcv}'")
        return self.data.head(row)
        

class stockchart:
    @staticmethod
    def plot(column, stock_data,stock_name):
        fig, ax = plt.subplots(figsize=(15, 10))
        try:
            mean = stock_data[column].mean()
            ax.plot(stock_data[column])
            ax.axhline(y=mean, color='red', label='Mean Line')
            
        except AttributeError:
            mean = stock_data.data[column].mean()
            ax.plot(stock_data.data[column])
            ax.axhline(y=mean, color='red', label='Mean Line')
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'Plot for {stock_name}({column.capitalize()})', fontsize=20)
        plt.legend()

    @staticmethod
    def plot_prices(column, stocks):
        fig, ax = plt.subplots(figsize=(15, 10))

        for stock_name, stock in stocks.items():
            try:
                ax.plot(stock.data[column], label=stock_name)
            except AttributeError:
                ax.plot(stock[column], label=stock_name)
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
        ax.set_xlabel('Date', fontsize=20)
        ax.set_ylabel('Price', fontsize=20)
        ax.set_title(f'Plot for Stock {column} Price', fontsize=20)
        plt.legend()
        return fig
    @staticmethod
    def plot_histplot(column, stock_data,stock_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            mean = stock_data.data[column].mean()
            sns.histplot(stock_data.data[column].values, kde=True, color="skyblue", ax=ax,alpha=0.8)
            ax.axvline(x=mean, c='red')
            
        except AttributeError:
            mean = stock_data[column].mean()
            sns.histplot(stock_data[column].values, kde=True, color="skyblue", ax=ax,alpha=0.8)
            ax.axvline(x=mean, c='red')
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
        
        ax.set_title(f'Histplot for {stock_name}({column.capitalize()})')
        ax.set_xlabel('Distribution return')
        ax.set_ylabel('frequency')
        plt.show()
        return fig
    @staticmethod
    def plot_scatter(column, stock_data,stock_name):
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        try:
            x = stock_data[column][1:]
            y = stock_data[column][:-1]
            plt.scatter(x=x,y=y,alpha=0.5)
        except AttributeError:
            x = stock_data.data[column][1:]
            y = stock_data.data[column][:-1]
            plt.scatter(x=x,y=y,alpha=0.5)
        except Exception as e:
            print(f"Lỗi không xác định: {e}")

        ax_scatter.set_title(f'Scatter Plot for {stock_name}({column.capitalize()})')
        ax_scatter.set_xlabel('value(t-1)')
        ax_scatter.set_ylabel('value(t)')
        plt.show()
        return fig_scatter


    @staticmethod
    def plot_ultimate(column, stocks):
        for stock_name, stock_data in stocks.items():
            stockchart.plot_histplot(column, stock_data,stock_name)
            stockchart.plot_scatter(column, stock_data,stock_name)
            stockchart.plot(column, stock_data,stock_name)
        if len(stocks) > 1:
            fig_all, ax_all = plt.subplots(figsize=(15, 10))
            for stock_name, stock in stocks.items():
                try:
                    stock.data[column].plot(kind='kde', label=stock_name, ax=ax_all)
                except AttributeError:
                    stock[column].plot(kind='kde', label=stock_name, ax=ax_all)
                except Exception as e:
                    print(f"Lỗi không xác định: {e}")
            ax_all.set_title('KDE Plot for All Stocks')
            ax_all.set_xlabel('Distribution return')
            ax_all.set_ylabel('frequency')
            ax_all.legend()
        plt.show()

        
class StockAnalysis:
    def __init__(self, data):
        self.data = data

    def calculate_moving_average(self, window):
        self.data[f'Moving_Average_{window}'] = self.data['close'].rolling(window=window).mean()

    def calculate_rsi(self, window):
        delta = self.data['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        average_gain = gain.rolling(window=window).mean()
        average_loss = loss.rolling(window=window).mean()

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))

        self.data[f'RSI_{window}'] = rsi

    def calculate_bollinger_bands(self, window):
        rolling_mean = self.data['close'].rolling(window=window).mean()
        rolling_std = self.data['close'].rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        self.data[f'Upper_Bollinger_Band_{window}'] = upper_band
        self.data[f'Lower_Bollinger_Band_{window}'] = lower_band

    def calculate_macd(self, short_window, long_window):
        short_ema = self.data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['close'].ewm(span=long_window, adjust=False).mean()

        macd = short_ema - long_ema
        signal_line = macd.ewm(span=9, adjust=False).mean()

        self.data[f'MACD_{short_window}_{long_window}'] = macd
        self.data[f'Signal_Line_{short_window}_{long_window}'] = signal_line
    def calculate_cumulative_returns(self):
        self.data['Cumulative_Returns'] = (1 + self.data['Daily_Returns']).cumprod() - 1
    def calculate_percent_daily_returns(self):
        self.data['Percent_Daily_Returns'] = self.data['close'].pct_change(1)
        self.data['Percent_Daily_Returns'].fillna(self.data['Daily_Returns'].mean())

    def calculate_daily_returns(self):
        self.data['Daily_Returns'] = np.log(self.data['close']/self.data['close'].shift(1))
        # mean  = self.data['Daily_Returns'].mean()
        # self.data['Daily_Returns'] = self.data['Daily_Returns'].fillna(mean)

    def calculate_sharpe_ratio(self, risk_free_rate):
        excess_returns = self.data['Daily_Returns'] - risk_free_rate
        average_excess_return = excess_returns.mean()
        std_dev_excess_return = excess_returns.std()
        sharpe_ratio = (average_excess_return / std_dev_excess_return) * np.sqrt(252)

        return sharpe_ratio

    def calculate_total_gains_and_losses(self):
        total_gains = self.data['Daily_Returns'][self.data['Daily_Returns'] > 0].sum()
        total_losses = self.data['Daily_Returns'][self.data['Daily_Returns'] < 0].sum()

        return total_gains, total_losses
    

def remove_outliers_IQR(data, column, threshold=1.5):
    data_copy = data.copy()
    Q1 = np.percentile(data_copy[column], 25)
    Q3 = np.percentile(data_copy[column], 75)
    IQR = Q3 - Q1
    # Tìm giá trị ngoại lai
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = (data_copy[column] < lower_bound) | (data_copy[column] > upper_bound)
    data_cleaned = data_copy.copy()

    data_cleaned.loc[outliers, column] = np.nan  # Gán giá trị ngoại lai bằng NaN
    print(f"Đã loại bỏ {outliers.sum()} phần tử")
    return data_cleaned


def test_stationarity(timeseries,name):

    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    plt.figure(figsize=(15, 10))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(f'Rolling Mean and Standard Deviation Of {name}')
    plt.show(block=False)
    
    print(f"Results of Argument dickey fuller test {name}:")
    adft = adfuller(timeseries,autolag='AIC')
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
def acf_pacf_plot(data,name = None):
    plt.figure(figsize = (15, 5))
    plot_acf(data,title=f'Autocorrelation: {name}')
    plt.figure(figsize = (15, 5))
    plot_pacf(data,title=f'Partial Autocorrelation: {name}')
 
   
# import pandas as pd
# from scipy import stats
# z_scores = stats.zscore(r_t)

# # Thiết lập ngưỡng Z-score
# threshold = 4

# # Tìm giá trị ngoại lai
# outliers = r_t[abs(z_scores) > threshold]
# print(f" Gía trị ngoại lai : {outliers}")

# data_cleaned = r_t[abs(z_scores) <= threshold]
# plt.figure(figsize=(8, 8))
# x=data_cleaned[1:]
# y=data_cleaned[:-1]
# plt.scatter(x=x,y=y)
# plt.title('Return rate')
# plt.xlabel('r(t-1)')
# plt.ylabel('r(t)')
# plt.show()