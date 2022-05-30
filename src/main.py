import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n = 20
import pandas as pd


def load_data():
    X_train = np.load("data/PEMS/X_train.npy")
    y_train = np.load("data/PEMS/y_train.npy")
    X_test = np.load("data/PEMS/X_test.npy")
    y_test = np.load("data/PEMS/y_test.npy")

    return X_train, y_train, X_test, y_test

def linearize_sensor(datapoint):
    new_datapoint = []
    for i in range(len(datapoint)):
        for j in range(len(datapoint[i])):
            new_datapoint.append(datapoint[i][j])
    return np.array(new_datapoint)

from src.data_analysis import *
from src.statistical_models import *

def main():

    day = np.load(file="data/PEMS/day.npy", allow_pickle=True)
    week = np.load(file="data/PEMS/week.npy", allow_pickle=True)
    data = week

    res = normality_test(data)
    print(res)
    res = one_sample_t_test(data)
    print(res)

    df = {"feature": data.tolist()}
    df = pd.DataFrame(df)

    X, y = [], []
    print(len(week))
    for i in range(len(data) - 21):
        X.append(week[i: i + 20])
        y.append(week[i + 20])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)

    regress_ar_auto_arima_model(y)
    holt_winters_model(X_train, y_train,X_test, y_test)

if __name__ == '__main__':
    main()