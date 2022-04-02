import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """

        # get path to csvs
        base_path = os.path.dirname(os.path.abspath(__file__))
        join_path = '../data/csv'
        csv_path = os.path.normpath(os.path.join(base_path, join_path))

        # create and return 'data' dictionary of dataframes
        csv_folder = os.listdir(csv_path)
        file_names = [file for file in csv_folder if file.endswith('.csv')]
        key_names = [file.strip('.csv').replace('_dataset', '').replace('olist_', '') for file in file_names]
        dfs = [pd.read_csv(os.path.join(csv_path, file)) for file in file_names]
        data = {key: df for (key, df) in zip(key_names, dfs)}
        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
