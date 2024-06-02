import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import imdlib as imd


def grab_heat_data():

    # Data download parameters
    variable = 'tmax'  # options are rain, tmin & tmax
    # start_day = "2015-06-01"  # earliest available data
    start_day = "2019-01-01"
    end_day = "2019-01-31"
    output_path = f'./dataset/heat/{variable}'

    # Call the API
    data = imd.get_real_data(variable, start_day, end_day, file_dir=output_path)

    return data


def open_heat_data():

    # Data download parameters
    variable = 'tmax'  # options are rain, tmin & tmax
    # start_day = "2015-06-01"  # earliest available data
    start_day = "2019-01-01"
    end_day = "2019-01-31"
    output_path = f'./dataset/heat/{variable}'

    # Call the API
    data = imd.open_real_data(variable, start_day, end_day, file_dir=output_path)

    return data


def explore_raw_data(data):
    ds = data.get_xarray()
    df = ds.to_dataframe()
    df_wide = df.unstack(level=1).unstack(level=1)
    # df_wide.columns = ['_'.join(map(str, col)).strip() for col in df_wide.columns]
    print(df_wide.index)
    print(df_wide.iloc[0])
    print(df_wide.columns)


def visualize_raw_data(data):
    ds = data.get_xarray()
    ds = ds.where(ds['tmax'] != -999.)  # Remove NaN values
    ds['tmax'].mean('time').plot()
    plt.show()


if __name__ == "__main__":
    # dataset = grab_heat_data()
    dataset = open_heat_data()
    explore_raw_data(dataset)
