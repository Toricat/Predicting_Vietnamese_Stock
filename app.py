import base64
from io import BytesIO
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from model.utils import SelectedStock, stockchart,StockAnalysis


app = Flask(__name__)

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
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
        stock_scatter = stockchart.plot_scatter('Daily_Returns',stock_calculate.data,"VCB")
        # stock_cleaned = remove_outliers_IQR(stock_calculate.data,"Daily_Returns",4)

        # has_nan = stock_cleaned["Daily_Returns"].isna()
        # stock_cleaned = stock_cleaned[~has_nan]

        # stock_scatter_clear = stockchart.plot_scatter('Daily_Returns',stock_cleaned,"Stock")
        #-------------------------------------------------------------------------------------------#

        return render_template('index.html', user_input=user_input, start_date=start_date, end_date=end_date,
                               
                               table_html=table_html,fig_open=fig_open,fig_close=fig_close,fig_volume=fig_volume,
                               
                               stock_calculate_data=stock_calculate_data,stock_sharpe_ratio=stock_sharpe_ratio,stock_gains=stock_gains,stock_losses=stock_losses,
                               
                               new_table_html=new_table_html,new_fig_open=new_fig_open,new_fig_close=new_fig_close,new_fig_volume=new_fig_volume,stock_calculate_describe=stock_calculate_describe
                               
                               ,stock_scatter=stock_scatter,stock_scatter_clear=stock_scatter_clear
                               )
    
    return render_template('index.html', user_input=None, start_date=None, end_date=None)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
