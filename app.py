import base64
from io import BytesIO
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arima.utils import SelectedStock, stockchart,StockAnalysis,acf_pacf_plot,visualization_result,remove_outliers_IQR,test_stationarity,evaluate_model
from arima.model import train_test_split,arima_model

app = Flask(__name__)

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig) 
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# Hàm mới để xử lý biểu đồ
def process_chart(stock):
    global_fig_base64 = fig_to_base64(stock)
    return global_fig_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        #-------------------------------------------------------------------------------------------#
        stock = SelectedStock(user_input, start_date, end_date)
        # Dataframe
        stock.data = stock.data.droplevel(level='Symbols', axis=1)
        table_html = stock.stock_infor().to_html(classes='table table-striped')
        # Gọi hàm plot_prices để tạo biểu đồ và nhận fig
        fig_open = stockchart.plot_prices('open', {'stock_data': stock.data})
        fig_close = stockchart.plot_prices('close', {'stock_data': stock.data})
        fig_volume = stockchart.plot_prices('volume', {'stock_data': stock.data})
        
        fig_open = process_chart(fig_open)
        fig_close = process_chart(fig_close)
        fig_volume = process_chart(fig_volume)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        #Xoa null
        stock.data.interpolate(method='quadratic', limit_direction='forward',limit_area='inside', axis=0, inplace=True)
        #New_Dataframe
        new_table_html = stock.stock_infor().to_html(classes='table table-striped')
        # Gọi hàm plot_prices để tạo biểu đồ và nhận fig
        new_fig_open = stockchart.plot_prices('open', {'stock_data': stock.data})
        new_fig_close = stockchart.plot_prices('close', {'stock_data': stock.data})
        new_fig_volume = stockchart.plot_prices('volume', {'stock_data': stock.data})
        
        new_fig_open = process_chart(new_fig_open)
        new_fig_close = process_chart(new_fig_close)
        new_fig_volume = process_chart(new_fig_volume)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        #Tinh toan
        stock_copy = stock.data.copy()
        stock_calculate = StockAnalysis(stock_copy)
        stock_calculate.calculate_moving_average(window=50)
        stock_calculate.calculate_rsi(window=14)
        stock_calculate.calculate_bollinger_bands(window=20)
        stock_calculate.calculate_macd(short_window=12, long_window=26)
        stock_calculate.calculate_daily_returns()
        # stock_calculate.calculate_percent_daily_returns()
        # stock_calculate.calculate_cumulative_returns()

        stock_sharpe_ratio = stock_calculate.calculate_sharpe_ratio(4)
        stock_gains,stock_losses=stock_calculate.calculate_total_gains_and_losses()
        
        stock_calculate.data = stock_calculate.data.iloc[1:]
        stock_calculate_data = stock_calculate.data.to_html(classes='table table-striped')
        
        stock_calculate_describe = stock_calculate.data.describe().to_html(classes='table table-striped')
        #-------------------------------------------------------------------------------------------#
         
        #-------------------------------------------------------------------------------------------#
        stock_scatter = stockchart.plot_scatter('Daily_Returns',stock_calculate.data,"Original data")
        stock_scatter = process_chart(stock_scatter)
        
        stock_cleaned,outliers = remove_outliers_IQR(stock_calculate.data,"Daily_Returns",4)

        has_nan = stock_cleaned["Daily_Returns"].isna()
        stock_cleaned = stock_cleaned[~has_nan]

        stock_scatter_clear = stockchart.plot_scatter('Daily_Returns',stock_cleaned,"Cleaned data")
        stock_scatter_clear = process_chart(stock_scatter_clear)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        stock_histplot_clear = stockchart.plot_histplot('Daily_Returns',stock_cleaned,user_input)
        stock_histplot_clear = process_chart(stock_histplot_clear)
        stock_plot_clear = stockchart.plot('Daily_Returns',stock_cleaned,user_input)
        stock_plot_clear = process_chart(stock_plot_clear)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        stationarity_result,stationarity_plot = test_stationarity(stock_cleaned['Daily_Returns'],user_input)
        stationarity_result = stationarity_result.to_frame().to_html(classes='table table-striped')
        stationarity_plot = process_chart(stationarity_plot)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        acf_plot ,pacf_plot = acf_pacf_plot(stock_cleaned['Daily_Returns'],user_input)
        acf_plot  = process_chart(acf_plot )
        pacf_plot  = process_chart(pacf_plot)
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        train_data, test_data =  train_test_split(stock_cleaned)
        train_arima = train_data['Daily_Returns']
        test_arima = test_data['Daily_Returns']
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        history = [x for x in train_arima]
        y = test_arima
        predictions,report_summary = arima_model(history,y, order=(1,1,0))   

        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        predictions_returns = visualization_result(test_data['Daily_Returns'],predictions,stock_cleaned['Daily_Returns'],title ="Returns",save="./static/img/arima_model_returns.pdf")
        predictions_returns  = process_chart( predictions_returns )
        result_returns_df = evaluate_model(test_data['Daily_Returns'], predictions)
        result_returns_df = result_returns_df.to_html(classes='table table-striped')
        #-------------------------------------------------------------------------------------------#
        
        #-------------------------------------------------------------------------------------------#
        date_minus_one_day = train_arima.index[-1]
        predictions_close = stock_cleaned["close"].loc[date_minus_one_day]*np.exp(np.cumsum(predictions))
        predictions_close_plot = visualization_result(test_data['close'],predictions_close,stock_cleaned['close'],title ="Close",save="./static/img/arima_model_close.pdf")
        predictions_close_plot  = process_chart( predictions_close_plot )
        result_close_df = evaluate_model(test_data['close'], predictions_close)
        result_close_df  = result_close_df.to_html(classes='table table-striped')
        #-------------------------------------------------------------------------------------------#
        
        plt.close('all')
        return render_template('index.html', user_input=user_input, start_date=start_date, end_date=end_date,
                               
                               table_html=table_html,fig_open=fig_open,fig_close=fig_close,fig_volume=fig_volume,
                               
                               stock_calculate_data=stock_calculate_data,stock_sharpe_ratio=stock_sharpe_ratio,stock_gains=stock_gains,stock_losses=stock_losses,
                               
                               new_table_html=new_table_html,new_fig_open=new_fig_open,new_fig_close=new_fig_close,new_fig_volume=new_fig_volume,stock_calculate_describe=stock_calculate_describe,
                               
                               stock_scatter=stock_scatter,stock_scatter_clear=stock_scatter_clear,outliers=outliers,
                               
                               stock_histplot_clear =stock_histplot_clear ,stock_plot_clear=stock_plot_clear,
                               
                               stationarity_result=stationarity_result,stationarity_plot=stationarity_plot,
                               
                               acf_plot = acf_plot ,pacf_plot = pacf_plot,
                               
                               report_summary=report_summary,
                               
                               predictions_returns= predictions_returns ,result_returns_df = result_returns_df,
                               
                               predictions_close_plot  = predictions_close_plot ,result_close_df = result_close_df
                               
                               )
    
    return render_template('index.html', user_input=None, start_date=None, end_date=None)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
