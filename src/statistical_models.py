from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.metrics import get_regress_perf_metrics

from src.visualizer import plot_actual_and_predicted_feature
from statsmodels.tsa.forecasting.theta import ThetaModel
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from causalimpact import CausalImpact
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from arch import arch_model
# import pyqt_fit.nonparam_regression as smooth
# from pyqt_fit import npr_methods
# from pyqt_fit import plot_fit
# import pyqt_fit.bootstrap as bs
from statsmodels.tsa.stattools import levinson_durbin as ld
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def acf_fn(data):
    plot_acf(data)
    plt.savefig("data/images/acf_plot.png")
    plt.show()

def pacf_fn(data):
    plot_pacf(data)
    plt.savefig("data/images/pacf_plot.png")
    plt.show()


def differentiate(data):
    pass


def random_walk_with_drift(data):
    pass

def non_param_regression(xs, ys):
    grid = np.r_[0:10:512j]
    k0 = smooth.NonParamRegression(xs, ys, method=npr_methods.SpatialAverage())
    k0.fit()
    plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
    plt.legend(loc='best')


def plot_residual_tests_spatial_average(xs, ys):
    k0 = smooth.NonParamRegression(xs, ys, method=npr_methods.SpatialAverage())
    yopts = k0(xs)
    res = ys - yopts
    plot_fit.plot_residual_tests(xs, yopts, res, 'Spatial Average')


def multiple_non_param_regressions(xs, ys):
    grid = np.r_[0:10:512j]


    def f(x):
        return 3 * np.cos(x / 2) + x ** 2 / 5 + 3

    k1 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=1))
    k2 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
    k3 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=3))
    k12 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=12))

    k1.fit()
    k2.fit()
    k3.fit()
    k12.fit()

    plt.figure()

    plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
    plt.plot(grid, k12(grid), 'b', label='polynom order 12', linewidth=2)
    plt.plot(grid, k3(grid), 'y', label='cubic', linewidth=2)
    plt.plot(grid, k2(grid), 'k', label='quadratic', linewidth=2)
    plt.plot(grid, k1(grid), 'g', label='linear', linewidth=2)
    plt.plot(grid, f(grid), 'r--', label='Target', linewidth=2)
    plt.legend(loc='best')


def plot_residual_tests_local_linear(xs, ys):
    k1 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=1))
    yopts = k1(xs)
    res = ys - yopts
    plot_fit.plot_residual_tests(xs, yopts, res, 'Local Linear')


def plot_residual_tests_local_quadratic(xs, ys):
    k2 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
    yopts = k2(xs)
    res = ys - yopts
    plot_fit.plot_residual_tests(xs, yopts, res, 'Local Quadratic')

def nonparam_regression_boostrap(xs, ys):
    grid = np.r_[0:10:512j]

    def fit(xs, ys):
        est = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
        est.fit()
        return est

    result = bs.bootstrap(fit, xs, ys, eval_points=grid, CI=(95, 99))

    plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
    plt.plot(grid, result.y_fit(grid), 'r', label="Fitted curve", linewidth=2)
    plt.plot(grid, result.CIs[0][0, 0], 'g--', label='95% CI', linewidth=2)
    plt.plot(grid, result.CIs[0][0, 1], 'g--', linewidth=2)
    plt.fill_between(grid, result.CIs[0][0, 0], result.CIs[0][0, 1], color='g', alpha=0.25)
    plt.legend(loc=0)


def durbin_levinson_algorithm(data, labels):
    model = ld(data, nlags=20)
    return model

def linear_regression(X_train, y_train, y_test):
    model_name = "linear regression"
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    X_test, y_pred = [X_train[-1]], [y_train[-1]]

    for i in range(20):
        prediction = lr.predict(np.reshape(X_test[0], (1, 20)))
        X_test.append(X_test[-1][1:] + [prediction])
        y_pred.append(prediction)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    logging_metrics_list = get_regress_perf_metrics(y_test,
                                                    y_pred,
                                                    model_name)

    print(logging_metrics_list)
    plot_actual_and_predicted_feature(y_test,
                                    y_pred,
                                    model_name)

    return logging_metrics_list, model_name


def log_results(logger, model_name, acc, conf_mat):
    logger.info(model_name)
    logger.info("Acc: {}: Conf mat:{}".format(acc, conf_mat))
    logger.info("-------------------------------")


def regress_ar_auto_arima_model(X, size:int=21):
    model_name = "Auto ARIMA-SARIMA Regression"
    train, test = X[:-size], X[-size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = SARIMAX(history, order=(4, 0, 18))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    logging_metrics_list = get_regress_perf_metrics(np.array(predictions[-21:]), np.array(history[-21:]),
                                                    model_name)
    plot_actual_and_predicted_feature(predictions[-21:], history[-21:], model_name)
    print(logging_metrics_list)
    return logging_metrics_list, model_name


def regress_arch_model(X_train, y_train, X_test, y_test):
    model_name = "ARCH Regression"
    n_test = len(X_test)
    model = arch_model(y_test, mean='Zero', vol='ARCH', p=15)
    model_fit = model.fit()
    y_pred = model_fit.forecast(horizon=n_test)
    final_y_pred = y_pred.variance.values[-1, :]
    plot_actual_and_predicted_feature(y_test, final_y_pred, model_name)
    logging_metrics_list = get_regress_perf_metrics(y_test, final_y_pred)
    return logging_metrics_list, model_name


def regress_garch_model(X_train, y_train, X_test, y_test):
    model_name = "GARCH Regression"
    n_test = len(y_test)
    model = arch_model(y_test, mean='Zero', vol='GARCH', p=15, q=15)
    model_fit = model.fit()
    y_pred = model_fit.forecast(horizon=n_test)
    final_y_pred = y_pred.variance.values[-1, :]

    plot_actual_and_predicted_feature(y_test, final_y_pred, model_name)

    logging_metrics_list = get_regress_perf_metrics(y_test, final_y_pred, model_name)
    return logging_metrics_list, model_name


def exponential_smoothing(X_train, y_train, X_test, y_test,
                          feature_scaler=None):
    model_name = "Simple Exponential Smoothing"
    model = SimpleExpSmoothing(y_train)

    fitted_model = model.fit(smoothing_level=0.2, optimized=False)

    y_pred = fitted_model.predict()

    print(len(y_test))
    print(len(y_pred))
    get_regress_perf_metrics(y_test, y_pred)
    if feature_scaler:
        y_pred = feature_scaler.inverse_transform(y_pred)
        y_test = feature_scaler.inverse_transform(y_test)
    plot_actual_and_predicted_feature(y_test, y_pred, model_name)
    logging_metrics_list = get_regress_perf_metrics(y_test, y_pred, model_name)

    return logging_metrics_list, model_name


def theta_model(X_train, y_train, X_test, y_test, feature_scaler=None):
    model_name = "Theta"

    model = ThetaModel(endog=y_train, period=len(y_train))
    model.fit()
    y_pred = model.predict()

    get_regress_perf_metrics(y_test, y_pred)
    if feature_scaler:
        y_pred = feature_scaler.inverse_transform(y_pred)
        y_test = feature_scaler.inverse_transform(y_test)
    plot_actual_and_predicted_feature(y_test, y_pred, model_name)

    logging_metrics_list = get_regress_perf_metrics(y_test, y_pred, model_name)

    return logging_metrics_list, model_name


def holt_winters_model(X_train, y_train, X_test, y_test, feature_scaler=None):
    model_name = "Exponential smoothing"

    fitted_model = ExponentialSmoothing(y_train, trend='mul', seasonal='mul', seasonal_periods=12).fit()
    y_pred = fitted_model.forecast(20)

    plot_actual_and_predicted_feature(y_test, y_pred, model_name)

    logging_metrics_list = get_regress_perf_metrics(np.array(y_test), np.array(y_pred), model_name)
