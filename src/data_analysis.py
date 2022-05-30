from scipy.stats import ttest_ind, kstest, normaltest, f_oneway, ttest_1samp
import numpy as np
import pandas as pd

from src.metrics import get_regress_perf_metrics


def normality_test(data):
    return normaltest(data)

def two_sample_t_test(a, b):
    return ttest_ind(a, b)

def ks_test(a, b):
    return kstest(a, b)

def quantiles(data, q:int=50):
    return np.quantile(data, q)

def one_sample_t_test(data, popmean: float=0.0):
    return ttest_1samp(data, popmean)

def one_way_anova(data):
    return f_oneway(data)


from statsmodels.tsa.stattools import adfuller

def is_data_stationary(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')

    if result[1] <= 0.05:
        return True
    else:
        return False

def transform_data_to_stationary(df, target_feature):
    max_shift_counter = 101
    for shift_counter in range(max_shift_counter):
        diff_arr = df[target_feature] - df[target_feature].shift(shift_counter)
        diff_arr = diff_arr.dropna()
        if is_data_stationary(diff_arr):
            df[target_feature] = diff_arr
            break
    return df[target_feature]

def deseazonalize(data):
    pass

def main():
    import numpy as np
    np.random.seed(1)

    l = 21
    steps = np.random.choice([-1, 1], size=l) + 0.05 * np.random.randn(l)
    position = np.cumsum(steps)
    y_pred = 0.05 * np.random.randn(l)
    import matplotlib.pyplot as plt
    week = np.load(file="data/PEMS/week.npy", allow_pickle=True)

    X, y = [], []
    print(len(week))
    for i in range(len(week) - 21):
        X.append(week[i: i + 20])
        y.append(week[i + 20])

    y_test = y[-21:]

    plt.plot(y_pred, label="random walk")
    plt.plot(y_test, label="original data")
    plt.legend()
    plt.savefig("data/images/random_walk.png")
    plt.show()

    logging_metrics_list = get_regress_perf_metrics(np.array(y_test),
                                                    np.array(y_pred),
                                                    "random_walk")
    print(logging_metrics_list)


if __name__ == '__main__':
    main()