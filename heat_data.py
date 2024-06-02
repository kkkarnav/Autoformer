import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
import imdlib as imd

warnings.filterwarnings("ignore")
OUTPUT_DIR = "./dataset/heat"


def grab_heat_data(variable):

    # Data download parameters
    output_path = f'{OUTPUT_DIR}/{variable}'

    # Earliest available 0.5x0.5 data is 2015 onwards
    data = imd.get_real_data(variable, "2015-06-01", "2015-12-31", file_dir=output_path)
    df = data.get_xarray().to_dataframe().unstack(level=1).unstack(level=1)
    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}_2015.csv")

    # Download the rest of the 0.5x0.5 data
    for year in range(2016, 2024):
        start_day = f"{year}-01-01"
        end_day = f"{year}-12-31"

        # Call the API and dump to file
        data = imd.get_real_data(variable, start_day, end_day, file_dir=output_path)
        df = data.get_xarray().to_dataframe().unstack(level=1).unstack(level=1)
        df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}_{year}.csv")


def read_heat_data(variable):

    df = pd.DataFrame()
    for year in range(2015, 2024):
        year_df = pd.read_csv(f"{OUTPUT_DIR}/0.5_{variable}_{year}.csv", header=[1, 2])[1:]\
            .reset_index(drop=True)\
            .replace(99.9000015258789, -99)
        df = pd.concat([df, year_df])

    df = df.reset_index(drop=True)
    count = (df == -99).sum()
    df = df.drop(columns=count[count > 100].index)
    df = df.reset_index(drop=True).replace(-99, 27)

    # print(f'(7.5 x 70.5) on {df.loc[0, ("lat", "lon")]}: {df.loc[0, ("7.5", "70.5")]} C')

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
    monthly_means = df.iloc[:, 1:].groupby(["year", "month"]).mean()
    ax = monthly_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Monthly means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India monthly min and max
    ax = monthly_means.max(axis=1).plot(color="red", alpha=0.5)
    ax = monthly_means.min(axis=1).plot(color="steelblue", alpha=0.5)
    ax = monthly_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Monthly min and max of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India daily mean
    ax = df.iloc[:, 1:-2].mean(axis=1).plot(color="firebrick")
    ax.set_xticks(df.index[::365])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India daily min and max
    ax = df.iloc[:, 1:-2].max(axis=1).plot(color="red", alpha=0.5)
    ax = df.iloc[:, 1:-2].min(axis=1).plot(color="steelblue", alpha=0.5)
    ax = df.iloc[:, 1:-2].mean(axis=1).plot(color="firebrick")
    ax.set_xticks(df.index[::365])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily min and max of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # Chunk of 100 grids
    ax = df.iloc[:, 1:101].mean(axis=1).plot(color="firebrick")
    ax.set_xticks(df.index[::365])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across 100 grids', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # A single grid
    ax = df.iloc[:, 1].plot(color="firebrick")
    ax.set_xticks(df.index[::365])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature of 8Nx77E', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()


def process_raw_data(dfmin, dfmax):

    dfmean = dfmin.copy()
    dfrange = dfmin.copy()

    for column in tqdm(range(1, len(dfmin.iloc[0]))):
        for row in range(len(dfmin)):
            dfmean.iloc[row, column] = (dfmax.iloc[row, column] + dfmin.iloc[row, column])/2
            dfrange.iloc[row, column] = dfmax.iloc[row, column] - dfmin.iloc[row, column]

    dfmean.to_csv(f"{OUTPUT_DIR}/0.5_tmean.csv")
    dfrange.to_csv(f"{OUTPUT_DIR}/0.5_trange.csv")
    return dfmean, dfrange


if __name__ == "__main__":

    # grab_heat_data("tmin")
    # grab_heat_data("tmax")
    tmin = read_heat_data("tmin")
    tmax = read_heat_data("tmax")

    # visualize_raw_data(tmin, "min.")
    # visualize_raw_data(tmax, "min.")

    tmean, trange = process_raw_data(tmin, tmax)

    pprint(tmean)
    pprint(trange)
    visualize_raw_data(tmin, "mean")
    visualize_raw_data(tmax, "range of")
