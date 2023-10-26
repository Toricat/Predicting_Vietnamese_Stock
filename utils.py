import pandas as pd
import numpy as np

import vnquant.data as dt
from vnquant import plot as pl
from vnquant import utils

import matplotlib.pyplot as plt
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
        
        
class MultiStockChart:
    def __init__(self, stocks):
        self.stocks = stocks
    def plot_prices(self, column):
        fig, ax = plt.subplots(figsize=(15, 10))

        for stock_name, stock in self.stocks.items():
            try:
                ax.plot(stock.data[column], label= stock_name)
            except AttributeError:
                ax.plot(stock[column], label= stock )
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Stock {column.capitalize()} Price')
        plt.legend()
    def plot_KDE(self,column):
        fig, ax = plt.subplots(figsize=(15, 10))
        for stock_name, stock in self.stocks.items():
            try:
                stock.data[column].plot(kind='kde', label=stock_name, ax=ax)
                plt.title('KDE Plot')
                plt.xlabel('Probability Distribution')
                plt.ylabel('Density')
                plt.show()
            except AttributeError:
                stock[column].plot(kind='kde', label=stock_name, ax=ax)
                plt.title('KDE Plot')
                plt.xlabel('Probability Distribution')
                plt.ylabel('Density')
                plt.show()
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
        fig, ax = plt.subplots(figsize=(15, 10))
        for stock_name, stock in self.stocks.items():
            try:
                stock.data[column].plot(kind='kde', label=stock_name, ax=ax)
            except AttributeError:
                stock[column].plot(kind='kde', label=stock_name, ax=ax)
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
        plt.title('KDE Plot')
        plt.xlabel('Probability Distribution')
        plt.ylabel('Density')
        plt.legend()
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
        self.data['Percent_Daily_Returns'] = self.data['close'].pct_change()
        self.data['Percent_Daily_Returns'] = self.data['Daily_Returns'].fillna(self.data['Daily_Returns'].mean())

    def calculate_daily_returns(self):
        self.data['Daily_Returns'] = np.log(self.data['close']/self.data['close'].shift(1))
        mean = self.data['Daily_Returns'].mean()
        self.data['Daily_Returns'].fillna(mean, inplace=True)
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