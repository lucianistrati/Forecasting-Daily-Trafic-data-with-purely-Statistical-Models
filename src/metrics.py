from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import check_array
from src.logger import empty_regress_loggings
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import matthews_corrcoef

EPSILON = 1e-10

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return _error(actual, predicted) / (actual + EPSILON)

def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def get_regress_perf_metrics(y_test, y_pred, model_name="",
                             target_feature="",
                             logging_metrics_list=empty_regress_loggings(),
                             visualize_metrics=False):
    if visualize_metrics:
        print("For " + model_name + " regression algorithm the following "
                                    "performance metrics were determined:")

    if target_feature == 'all':
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == "MSE":
            logging_metrics_list[i][1] = str(mean_squared_error(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAE":
            logging_metrics_list[i][1] = str(mean_absolute_error(y_test,
                                                                 y_pred))
        elif logging_metrics_list[i][0] == "R2":
            logging_metrics_list[i][1] = str(r2_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAPE":
            logging_metrics_list[i][1] = str(mape(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MDA":
            logging_metrics_list[i][1] = str(mda(y_test, y_pred))
        elif logging_metrics_list[i][0] == "MAD":
            logging_metrics_list[i][1] = 0.0

    if visualize_metrics:
        print("MSE: ", mean_squared_error(y_test, y_pred))
        print("MAE: ", mean_absolute_error(y_test, y_pred))
        print("R squared score: ", r2_score(y_test, y_pred))
        print("Mean absolute percentage error:", mape(y_test, y_pred))
        try:
            print("Mean directional accuracy:", mda(y_test, y_pred))
        except TypeError:
            print("Type error", model_name)

    return logging_metrics_list


def main():
    a = np.random.rand(1, 100)
    b = np.random.rand(1, 100)

    logging_metrics_list = get_regress_perf_metrics(a, b)

    print(logging_metrics_list)

if __name__=='__main__':
    main()
