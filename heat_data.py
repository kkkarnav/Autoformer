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
    df = df.drop(columns=count[count > 30].index)
    df = df.reset_index(drop=True).replace(-99, 26.98989)

    # print(f'(7.5 x 70.5) on {df.loc[0, ("lat", "lon")]}: {df.loc[0, ("7.5", "70.5")]} C')

    return df


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
    ax.set_xticks(df.index[::366])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India daily min and max
    ax = df.iloc[:, 1:-2].max(axis=1).plot(color="red", alpha=0.5)
    ax = df.iloc[:, 1:-2].min(axis=1).plot(color="steelblue", alpha=0.5)
    ax = df.iloc[:, 1:-2].mean(axis=1).plot(color="firebrick")
    ax.set_xticks(df.index[::366])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily min and max of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # Chunk of 100 grids
    ax = df.iloc[:, 1:101].mean(axis=1).plot(color="firebrick")
    ax.set_xticks(df.index[::366])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across 100 grids', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # A single grid
    ax = df.iloc[:, 1].plot(color="firebrick")
    ax.set_xticks(df.index[::366])
    ax.set_xticklabels(df.iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature of 8Nx77E', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()


def convert_to_format(df):

    # Melt the df
    df = df.melt(id_vars=[("lat", "lon")], value_name="TEMP")
    df = df.rename(columns={("lat", "lon"): "date", "variable_0": "LAT", "variable_1": "LONG"})

    # Sort it
    df["LAT"] = df["LAT"].apply(lambda x: float(x))
    df["LONG"] = df["LONG"].apply(lambda x: float(x))
    df = df.sort_values(by=["LAT", "LONG", "date"]).reset_index(drop=True)

    # Shift the value column to second
    cols = list(df.columns)
    df = df[[cols[0], cols[-1]] + cols[1:-1]]

    pprint(df)

    df.to_csv(f"{OUTPUT_DIR}/0.5_tmean_india.csv", index=False)
    return df


if __name__ == "__main__":

    # Get raw tmin and tmax data as GRD files from IMDlib
    # grab_heat_data("tmin")
    # grab_heat_data("tmax")

    # Read raw tmin and tmax data into csv format
    tmin = read_heat_data("tmin")
    tmax = read_heat_data("tmax")

    # Visualize temporal trends in min and max data
    # visualize_raw_data(tmin, "min.")
    # visualize_raw_data(tmax, "min.")

    # Convert tmin and tmax to the mean temperature per day
    tmean, trange = process_raw_data(tmin, tmax)

    # Read the processed tmean data
    tmean = pd.read_csv(f"{OUTPUT_DIR}/0.5_tmean.csv", header=[0, 1], index_col=0)
    trange = pd.read_csv(f"{OUTPUT_DIR}/0.5_trange.csv", header=[0, 1], index_col=0)

    # pprint(tmean)
    # pprint(trange)
    # visualize_raw_data(tmean, "mean")
    # visualize_raw_data(trange, "range of")

    tmeanf = convert_to_format(tmean)
