import matplotlib.pyplot as plt
from io import BytesIO
import base64

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# Hàm mới để xử lý biểu đồ
def process_chart(stock):
    global_fig_base64 = fig_to_base64(stock)
    return global_fig_base64