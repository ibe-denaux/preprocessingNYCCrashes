import numpy as np
import pandas as pd
from cleaning_functions import *

path = 'assets\data_100000.csv'
crashes = pd.read_csv(path, parse_dates=['crash_date', 'crash_time'])

pd.set_option('display.max_columns', None)

# check duplicate rows: False
# print("duplicates:", crashes.duplicated().any())
# print(crashes.isnull().sum())
# print("sum:", crashes.duplicated().sum())
# print(crashes[crashes.duplicated()].sort_values(by='crash_date'))


# DROPPING UNNECESSARY COLUMNS
drop_columns_redundant(crashes)
drop_columns_not_needed_for_machine_learning(crashes)

# check whole rows for null
crashes.dropna(how='all')  # no rows deleted so now rows are all empty
print(crashes.isnull().sum())

# consolidating strings
strip_strings_in_whole_dataset(crashes)
strings_to_lower_in_whole_dataset(crashes)

# GET TO KNOW THE DATASET
# print(crashes.head())
# print(crashes.columns)
# print(crashes.dtypes)
# crashes.info()


# convert datatypes
unpack_date_and_time_items(crashes)

# 169 rows have values 0.0 for latitude and longitude
drop_rows_without_longitude_latitude_streetname(crashes)  # -51 rows
fill_rows_with_streetname_without_longlati(crashes)  # fill 114 rows
drop_rows_without_replacements_for_longlati(crashes)  # drop 4

# 4x strange -201 longitude value (East of Japan). For a bridge.
# I found the right value on the internet (it is not in the dataset)
fill_longitude_where_wrong(crashes)

# also: ROWS WITHOUT MISSING NAN VALUES
fill_in_nan_for_latitude_longitude(crashes)

# ANOTHER LOOK AT DUPLICATES. We dropped columns so new duplicate rows who were initially not duplicates
# 124: drop them for ease? Or are they not duplicates?
# print("duplicates:", crashes.duplicated().any())
# print("sum:", crashes.duplicated().sum())
# print(crashes[crashes.duplicated()].sort_values(by=['year', 'month', 'day', 'hours', 'minutes']))
# crashes.drop_duplicates(inplace=True)
# print(crashes.shape)

# check null values
# print(crashes.isnull().sum())


# print(crashes['vehicle_type_code1'].unique())
# print(crashes['contributing_factor_vehicle_1'].unique())


# print(crashes.isnull().sum())
# print(crashes.shape)


# ONT-HOT-ENCODING
# ATV, bicycle, car/suv, ebike, escooter, truck/bus, motorcycle, other)
# bike = ['bike', ]
# car = ['sedan', 'station', 'taxi', 'pick', 'van', 'deliv',
#        'convertible', 'e-350', 'passenger vehicle', 'cab',   ]
# big_truck_bus = ['truck', 'bus', 'tanker', 'flat', 'rv', 'garbage' ,
#              'concrete_mixer', 'tract', 'mack' ]
# other = ['moped', 'pas', 'lift noom']

# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['delivery v', 'delv'], "van")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['escooter', 'e-sco', 'e-scoter', 'motor',
#                                                                        'scooter', 'motorbike', 'motorcycle',
#                                                                        'e-scooter', 'motorscooter', ], "motorbike")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['e bike', 'e-bik', 'ebike', 'e bik', 'e-bike',
#                                                                        'moped', 'bike'], "bike")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(
#     ['bus', 'taxi', 'schoolbus', 'fire', 'fire truck'], "public vehicles")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['ambul', 'ambulance'], "service vehicles")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['truck', 'tanker', 'flat bed',
#                                                                        'rv', 'garbage', 'concrete_mixer', 'tract',
#                                                                        'mack', 'box truck', 'tractor truck diesel',
#                                                                        'dump', 'garbage or refuse', 'carry all',
#                                                                        'tractor truck gasoline', 'tow truck / wrecker',
#                                                                        'chassis cab',
#                                                                        'tanker', 'refrigerated van', 'pick-up truck',
#                                                                        'multi-wheeled vehicle',
#                                                                        'beverage truck', 'armored truck'
#                                                                        ], "big_trucks")
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(
#     ['taxi', 'sedan', 'station wagon/sport utility vehicle',
#      'convertible', 'pk', '4 dr sedan', 'passenger vehicle', 'limo', 'sport utility / station wagon',
#      '3-door'], 'car')
#
# crashes["vehicle_type_code1"] = crashes["vehicle_type_code1"].replace(['trailer', 'trail', 'stake or rack'],
#                                                                       'car-attachment')
#



print(len(crashes.vehicle_type_code1.unique()))

crashes_copy = crashes.copy()

# print(crashes_copy['vehicle_type_code1'].value_counts().head(50))
# crashes_copy
# for idx, row in crashes_copy.iterrows():
#     if row['vehicle_type_code1']:
#         print(crashes_copy.loc[idx, 'vehicle_type_code1')
# if 'car' in vehicle:
#     crash
categories = ['car',
              'big_trucks',
              'van',
              'bike',
              'motorbike',
              "service vehicles",
              "public vehicles",
              'vehicle-attachment']  # losing rows
condition_categories_one = crashes_copy['vehicle_type_code1'].isin(categories)
condition_categories_two = crashes_copy['vehicle_type_code2'].isin(categories)
condition_categories_three = crashes_copy['vehicle_type_code_3'].isin(categories)
condition_categories_four = crashes_copy['vehicle_type_code_4'].isin(categories)
condition_categories_five = crashes_copy['vehicle_type_code_5'].isin(categories)

# print('condition cats', condition_categories.value_counts())
df_categories_one = crashes_copy[condition_categories_one]
df_categories_two = crashes_copy[condition_categories_two]
df_categories_three = crashes_copy[condition_categories_three]
df_categories_four = crashes_copy[condition_categories_four]
df_categories_five = crashes_copy[condition_categories_five]

df_categories_vehicles = pd.concat([df_categories_two,
                                   df_categories_two,
                                   df_categories_three,
                                   df_categories_four,
                                   df_categories_five])


into_categories(df_categories_vehicles, 'vehicle_type_code1')
into_categories(df_categories_vehicles, 'vehicle_type_code2')
into_categories(df_categories_vehicles, 'vehicle_type_code_3')
into_categories(df_categories_vehicles, 'vehicle_type_code_4')
into_categories(df_categories_vehicles, 'vehicle_type_code_5')

# print(df_categories_vehicles)
# print(df_categories.value_counts())

dummies = pd.get_dummies(data=df_categories, columns=['vehicle_type_code1',
                                                      'vehicle_type_code2',
                                                      'vehicle_type_code_3',
                                                      'vehicle_type_code_4',
                                                      'vehicle_type_code_5'])
print(dummies.head())
