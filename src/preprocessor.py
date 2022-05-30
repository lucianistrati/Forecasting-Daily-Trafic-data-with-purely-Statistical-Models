import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_heatmap(y_pred, y_test):
    cf_matrix = confusion_matrix (y_pred, y_test)
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
    sns.heatmap(cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
    plt.show()

def get_scaler(scaling_option="standard"):
    if scaling_option == "standard":
        return StandardScaler()
    elif scaling_option == "minmax":
        return MinMaxScaler()
    else:
        raise Exception("Wrong scaling option given!")


def load_dim_reducer(dim_red_option, n_components):
    if dim_red_option == "PCA":
        dim_reducer = PCA(n_components=n_components)
    elif dim_red_option == "TSNE":
        dim_reducer = TSNE(n_components=n_components)
    elif dim_red_option == "FA":
        dim_reducer = FactorAnalysis(n_components=n_components)
    elif dim_red_option == "SVD":
        dim_reducer = TruncatedSVD(n_components=n_components)
    else:
        raise Exception("wrong dimensionality reduction option given!")
    return dim_reducer


def get_class_weight(labels):
    class_weight_dict = dict()
    for label in labels:
        if label not in class_weight_dict.keys():
            class_weight_dict[label] = 1
        else:
            class_weight_dict[label] += 1
    num_labels = len(labels)
    for (key, value) in class_weight_dict.items():
        class_weight_dict[key] = num_labels / class_weight_dict[key]
    return class_weight_dict

def merge_train_and_val_data(X_train, y_train, X_validation, y_validation,
                             X_test, y_test):
    return np.concatenate((X_train, X_validation)), \
           np.concatenate((y_train, y_validation)), np.array(X_test), \
           np.array(y_test)

def convert_to_numpy_arrays(X_train, y_train, X_validation, y_validation,
                            X_test, y_test):
    return np.array(X_train), np.array(y_train), np.array(
        X_validation), np.array(y_validation), np.array(X_test), np.array(
        y_test)

def convert_row_to_float(row):
    keys = list(row.keys())
    for i in range(len(keys)):
        if isinstance(row[keys[i]], str):
            row[keys[i]] = float(row[keys[i]])
    return row


def convert_df_to_float(df):
    for i in range(len(df)):
        row = df.iloc[i]
        df.iloc[i] = convert_row_to_float(row)
    return df
