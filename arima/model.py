import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

def train_test_split(data):
    train_data, test_data = data[0:int(len(data)*0.9)], data[int(len(data)*0.9):]
    return train_data, test_data
def arima_model(history,y, order=(1,1,0)):
    predictions = list()
    for i in tqdm(range(0, len(y))):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        result = model_fit.forecast()[0]
        predictions.append(result)
        obs = y[i]
        history.append(obs)
    report_model = model_fit.summary()
    return predictions,report_model