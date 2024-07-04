import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

warnings.filterwarnings("ignore")
OUTPUT_DIR = "./dataset/heat_main"


def read_heat_data(variable):
    df = pd.DataFrame()

    df = pd.read_csv(f"{OUTPUT_DIR}/main_{variable}.csv", header=[1, 2])[1:] \
        .reset_index(drop=True) \
        .replace(99.9000015258789, -99)

    count = (df == -99).sum()
    df = df.drop(columns=count[count > 40].index)
    df = df.reset_index(drop=True).replace(-99, 25.98989)

    return df


def read_rt_heat_data(variable):
    df = pd.DataFrame()
    for year in range(2015, 2024):
        year_df = pd.read_csv(f"{OUTPUT_DIR}/{variable}/0.5_{variable}_{year}.csv", header=[1, 2])[1:] \
            .reset_index(drop=True) \
            .replace(99.9000015258789, -99)
        df = pd.concat([df, year_df])

    df = df.reset_index(drop=True)
    count = (df == -99).sum()
    df = df.drop(columns=count[count > 30].index)
    df = df.reset_index(drop=True).replace(-99, 26.98989)

    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}.csv")
    return df


def visualize_raw_data(df, label):
    # India annual mean
    df["year"] = pd.to_datetime(df.iloc[:, 0]).apply(lambda x: x.year)
    annual_means = df.iloc[:, 1:].groupby("year").mean()
    ax = annual_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Annual means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India annual min and max
    ax = annual_means.max(axis=1).plot(color="red", alpha=0.5)
    ax = annual_means.min(axis=1).plot(color="steelblue", alpha=0.5)
    ax = annual_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Annual min and max of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India monthly mean
    df["month"] = pd.to_datetime(df.iloc[:, 0]).apply(lambda x: x.month)
    monthly_means = df.iloc[:, 1:].groupby(["year", "month"]).mean().iloc[-240:]
    ax = monthly_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Monthly means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India daily mean
    ax = df.iloc[:, 1:-2].mean(axis=1).iloc[-1825:].plot(color="firebrick")
    ax.set_xticks(df.iloc[-1825:].index[::366])
    ax.set_xticklabels(df.iloc[-1825:].iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

def aggregate_ts(df):


if __name__ == "__main__":
    # Read raw tmin and tmax data into csv format
    tmax = read_heat_data("tmax")
    tmax.index = pd.to_datetime(tmax[("lat", "lon")])
    # Visualize temporal trends in min and max data
    # visualize_raw_data(tmax, "max.")

    tmax[("avg", "avg")] = tmax.iloc[:, 1:].mean(axis=1)
    print(tmax.head())
    print(tmax[("avg", "avg")])

    """result = seasonal_decompose(tmax[('avg', 'avg')], model='additive')
    result.plot()
    plt.show()

    result2 = seasonal_decompose(tmax.iloc[:, -1], model='multiplicative')
    result2.plot()
    plt.show()"""

    from statsmodels.tsa.seasonal import STL

    # Perform STL decomposition
    stl_result = STL(tmax.iloc[:, -1], seasonal=13).fit()
    # Extract the components
    seasonal_stl = stl_result.seasonal
    trend_stl = stl_result.trend
    residual_stl = stl_result.resid
    # Visualize the components
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(tmax.iloc[:, -1], label='Original Data')
    plt.title('Original Time Series')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(trend_stl, label='Trend (STL)', color='orange')
    plt.title('Trend Component (STL)')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(seasonal_stl, label='Seasonal (STL)', color='green')
    plt.title('Seasonal Component (STL)')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(residual_stl, label='Residual (STL)', color='red')
    plt.title('Residual Component (STL)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    from statsmodels.tsa.seasonal import seasonal_decompose

    # Perform multiplicative decomposition
    result = seasonal_decompose(tmax.iloc[:, -1], model='multiplicative')
    # Extract the components
    trend_mul = result.trend.dropna()
    seasonal_mul = result.seasonal.dropna()
    residual_mul = result.resid.dropna()
    # Visualize the components
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(tmax.iloc[:, -1], label='Original Data')
    plt.title('Original Time Series')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(trend_mul, label='Trend (Multiplicative)', color='orange')
    plt.title('Trend Component (Multiplicative)')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(seasonal_mul, label='Seasonal (Multiplicative)', color='green')
    plt.title('Seasonal Component (Multiplicative)')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(residual_mul, label='Residual (Multiplicative)', color='red')
    plt.title('Residual Component (Multiplicative)')
    plt.legend()
    plt.tight_layout()
    plt.show()
