import numpy as np
import pandas as pd
from cleaning_functions import *

path = 'assets\data_100000.csv'
crashes = pd.read_csv(path, parse_dates=['crash_date', 'crash_time'])
crashes_copy = crashes.copy()
pd.set_option('display.max_columns', None)


# DROPPING UNNECESSARY COLUMNS
drop_columns_redundant(crashes_copy)
drop_columns_not_needed_for_machine_learning(crashes_copy)

# check whole rows for null
crashes_copy.dropna(how='all')  # no rows deleted so now rows are all empty
# print(crashes_copy.isnull().sum())

# consolidating strings
strip_strings_in_whole_dataset(crashes_copy)
strings_to_lower_in_whole_dataset(crashes_copy)

# convert datatypes
unpack_date_and_time_items(crashes_copy)

# 169 rows have values 0.0 for latitude and longitude
drop_rows_without_longitude_latitude_streetname(crashes_copy)  # -51 rows
fill_rows_with_streetname_without_longlati(crashes_copy)  # fill 114 rows
drop_rows_without_replacements_for_longlati(crashes_copy)  # drop 4

fill_longitude_where_wrong(crashes_copy)
fill_in_nan_for_latitude_longitude(crashes_copy)

crashes_copy = one_hot_encoding_vehicles(crashes_copy)


crashes_copy.drop(columns='on_street_name', inplace=True)
crashes_copy.to_csv('crashes_newyork.csv', index=False)



print(crashes_copy.columns)
print(crashes_copy.shape)
print(crashes_copy.dtypes)
print(crashes_copy.head())