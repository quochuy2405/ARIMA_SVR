import datetime as dt
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

# List of column names
field_names = ['formatted_date', 'high', 'low', 'open', 'close', 'volume', 'adjclose']

cols_y_close = ['close']
cols_y_open = ['open']
cols_y_high = ['high']
cols_y_low = ['low']

def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def load_error_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# Prepare variables
def predict(model, error_model, last_value):
    predict_price = model.predict(n_periods=1)
    predict_error = error_model.predict(np.array([[predict_price.values[0]]]))[0]
    return predict_price.values[0] + predict_error

# Load dataframe
df = pd.read_csv("./data/clean/btc.csv")
df['formatted_date'] = pd.to_datetime(df['formatted_date'])
df.set_index("formatted_date", inplace=True)

# Load models
model_close = load_model("./models/close/arima_model.pkl")
model_open = load_model("./models/open/arima_model.pkl")
model_high = load_model("./models/high/arima_model.pkl")
model_low = load_model("./models/low/arima_model.pkl")

model_error_close = load_error_model("./models/close/svr_model.pkl")
model_error_open = load_error_model("./models/open/svr_model.pkl")
model_error_high = load_error_model("./models/high/svr_model.pkl")
model_error_low = load_error_model("./models/low/svr_model.pkl")

n_periods = 1
for i in range(n_periods):
    next_day = (dt.date.today() + dt.timedelta(days=i + 1)).strftime("%Y-%m-%d")
    row = [
        next_day,
        predict(model_high, model_error_high, df['high'].values[-1])[0],
        predict(model_low, model_error_low, df['low'].values[-1])[0],
        predict(model_open, model_error_open, df['open'].values[-1])[0],
        predict(model_close, model_error_close, df['close'].values[-1])[0],
        0,
        0
    ]
    df_pred = pd.DataFrame([row], columns=field_names)
    df = pd.concat([df, df_pred])

df.to_csv('final_pred_arima_lstm.csv', index=False)
