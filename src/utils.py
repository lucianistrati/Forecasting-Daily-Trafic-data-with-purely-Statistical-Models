from src.preprocessor import convert_row_to_float, convert_df_to_float, convert_to_numpy_arrays
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import datetime
import calendar

X_train, y_train, X_validation, y_validation, X_test, y_test = [], [], \
                                                                   [], [], \
                                                                   [], []
closing_prices = []

def reinitialize_data():
    global X_train, y_train, X_validation, y_validation, X_test, y_test, \
        closing_prices
    X_train, y_train, X_validation, y_validation, X_test, y_test = [], [], \
                                                                   [], [], \
                                                                   [], []
    closing_prices = []


reinitialize_data()
X, y = [], []
final_data_point = None

TRAIN_PCT = 0.6
VAL_PCT = 0.2
TEST_PCT = 0.2

from statsmodels.tsa.stattools import adfuller


# Store in a function for later use!
def is_data_stationary(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:', result)
    if result[1] <= 0.05:
        return True
    else:
        return False

def transform_data_to_stationary(df, target_feature):
    max_shift_counter = 20
    for shift_counter in range(max_shift_counter):
        diff_arr = df[target_feature] - df[target_feature].shift(shift_counter)
        diff_arr = diff_arr.dropna()
        if is_data_stationary(diff_arr):
            df[target_feature] = diff_arr
            break
    return df[target_feature]

