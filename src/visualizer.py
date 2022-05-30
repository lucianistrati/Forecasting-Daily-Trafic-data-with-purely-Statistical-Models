import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def plot_actual_and_predicted_feature(actual_price, predicted_price, model_name,
                                    date_interval=None, preview=True):
    if not date_interval:
        date_interval = range(1, len(actual_price) + 1)
    plt.plot(date_interval, actual_price, "g", label = "Actual feature")
    plt.plot(date_interval, predicted_price, "b", label = "Predicted feature")
    plt.legend()
    plt.title(model_name)
    plt.savefig(f"data/images/{model_name}.png")
    if preview:
        plt.show()


def boxplot(X, show: bool=False):
    fig = plt.figure()
    plt.boxplot(X)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    if show:
        plt.show()


from collections import Counter

def plot_barchart(values, problem):
    cnt = Counter(values)
    xs = list(range(1, max(values) + 1))
    ys = [cnt[x] for x in xs]
    plt.bar(xs, ys, align='center')
    plt.savefig(f"data/images/barchart_{problem}.png")
    plt.show()

from typing import List, Tuple

def plot_piechart(sizes: List[float], labels: List[str]=None, explode: Tuple[float]=None):
    sizes = list(sizes)
    counter = Counter(sizes)

    items = list(counter.items())
    labels = [item[0] for item in items]
    sizes = [item[1] for item in items]

    if explode is None:
        max_val = sizes.index(max(sizes))
        explode = [0 for _ in range(len(sizes))]
        explode[max_val] = 0.25
        explode = tuple(explode)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    plt.savefig("data/images/piechart.png")
    plt.show()


def plot_area_chart(df):
    plt.stackplot(df.index,
                  [df['original'] - df['original'].rolling(5).mean(),
                   df['original'] - df['original'].rolling(10).mean(),
                   df['original'] - df['original'].rolling(20).mean()],
                  labels=['5-SMA-difference-delay', '10-SMA-difference-delay', '20-SMA-difference-delay'],
                  alpha=0.8)

    plt.legend(loc='best')

    plt.savefig("data/images/area_chart.png")
    plt.show()


def moving_averages_plot(df):
    plt.plot(df['original'], label='Close')
    plt.plot(df['original'].rolling(5).mean(), label='5-SMA')
    plt.plot(df['original'].rolling(10).mean(), label='10-SMA')
    plt.plot(df['original'].rolling(20).mean(), label='20-SMA')

    plt.legend(loc='best')

    plt.savefig("data/images/moving_averages.png")
    plt.show()
