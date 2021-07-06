import pandas as pd
import numpy as np


def drop_columns_redundant(df):
    df.drop(columns=['location',
                     'borough',
                     'zip_code',
                     'off_street_name',
                     'cross_street_name'
                     ], inplace=True)


def drop_columns_not_needed_for_machine_learning(df):
    df.drop(columns=['collision_id',
                     # 'contributing_factor_vehicle_1',
                     'contributing_factor_vehicle_2',
                     'contributing_factor_vehicle_3',
                     'contributing_factor_vehicle_4',
                     'contributing_factor_vehicle_5',
                     # 'vehicle_type_code1',
                     # 'vehicle_type_code2',
                     # 'vehicle_type_code_3',
                     # 'vehicle_type_code_4',
                     # 'vehicle_type_code_5',
                     ], inplace=True)


def strip_strings_in_whole_dataset(df):
    for column in df.columns:
        try:
            df[column] = df[column].str.strip()
        except:
            continue


def strings_to_lower_in_whole_dataset(df):
    for column in df.columns:
        try:
            df[column] = df[column].str.lower()
        except:
            continue


def unpack_date_and_time_items(df):
    # columns crash_date & crash_time have been parsed as datetimes in read_csv
    df['year'] = df.crash_date.dt.year
    df['month'] = df.crash_date.dt.month
    df['day'] = df.crash_date.dt.day
    df['hours'] = df.crash_time.dt.hour
    df['minutes'] = df.crash_time.dt.minute
    df.drop(columns=['crash_date', 'crash_time'], inplace=True)


def drop_rows_without_longitude_latitude_streetname(df):
    # make new df with null or 0.0 for three columns
    condition_long = df.longitude == 0.0
    condition_lati = df.latitude == 0.0
    condition_name = df.on_street_name.isnull()

    df_to_drop = df[condition_long &
                    condition_lati &
                    condition_name]

    drop_rows_from_df(df, df_to_drop)

    # test_if_dropped = df[condition_long &
    #                      condition_lati &
    #                      condition_name]
    # print(test_if_dropped.size)


def fill_rows_with_streetname_without_longlati(df):
    # 118 rows with streetname but no coordinates. Collect them in a df to inspect:
    # df_streetname_without_longitude_latitude = df[df['longitude'] > -70]

    # collect coordinates for all locations
    df_coordinates = df[(df.longitude < -70) &
                        (df.longitude > -75) &
                        (df.latitude > 35) &
                        (df.latitude < 45)]
    df_coordinates = df_coordinates.dropna(subset=['on_street_name'])
    df_coordinates = df_coordinates.drop_duplicates()
    df_coordinates['on_street_name'] = df_coordinates['on_street_name'].str.strip()
    # print(df_coordinates.shape)

    # check if coordinates for street names are available. If so, fill them in in df
    list_of_street_names_with_coordinates = df_coordinates.on_street_name.unique()
    for idx, row in df.iterrows():
        if row['longitude'] == 0.0 and row['latitude'] == 0.0:
            street_name = row['on_street_name'].strip()
            if street_name in list_of_street_names_with_coordinates:
                street_name_data_row = df_coordinates[df_coordinates.on_street_name == street_name].head(1)
                long = street_name_data_row['longitude'].iloc[0]  # take first longitude from first row
                df.loc[idx, 'longitude'] = long
                lati = street_name_data_row['latitude'].iloc[0]  # take first latitude from first row
                df.loc[idx, 'latitude'] = lati


def drop_rows_without_replacements_for_longlati(df):
    # identify rows
    remaining_rows_no_coordinates = df[df['longitude'] > -70]
    drop_rows_from_df(df, remaining_rows_no_coordinates)

    # check if empty
    # remaining_rows_no_coordinates = df[df['longitude'] > -70]
    # print(remaining_rows_no_coordinates.shape)


def drop_rows_from_df(df, df_rows_to_drop):
    for index, row in df.iterrows():
        if index in df_rows_to_drop.index:
            df.drop(index, inplace=True)


def fill_longitude_where_wrong(df):
    # print(crashes[crashes['longitude']<-75]['longitude'])
    for idx, row in df.iterrows():
        if row.longitude < -75:
            df.loc[idx, 'longitude'] = -73.95429571247779

    # print(crashes.iloc[5804])
    # print(crashes[crashes.on_street_name == 'queensboro bridge upper roadway'])


def fill_in_nan_for_latitude_longitude(crashes):
    # these are bridges, beaches, parks, tunnels, ...
    # I managed to replace nan for longlati for nearly 6000 rows
    # drop the other rows: no longlati (name nan or name not in set with knows coordinates)
    condition_long = crashes['longitude'].isnull()
    condition_lati = crashes['latitude'].isnull()

    df_coordinates = crashes[(crashes.longitude < -70) &
                             (crashes.longitude > -75) &
                             (crashes.latitude > 35) &
                             (crashes.latitude < 45)]
    df_coordinates = df_coordinates.dropna(subset=['on_street_name'])
    df_coordinates = df_coordinates.drop_duplicates()
    df_coordinates['on_street_name'] = df_coordinates['on_street_name'].str.strip()

    list_of_streetnames_with_coordinates = df_coordinates.on_street_name.unique()

    print(crashes.on_street_name.unique())
    for idx, row in crashes.iterrows():
        if not (-75 < row['longitude'] < -70):  # nan
            if not (38 < row['latitude'] < 42):  # nan
                if row['on_street_name'] is not np.nan:  # streetnames
                    streetname = row['on_street_name'].strip()
                    if streetname in list_of_streetnames_with_coordinates:
                        streetname_data_row = df_coordinates[df_coordinates.on_street_name == streetname].head(1)
                        long = streetname_data_row['longitude'].iloc[0]
                        lati = streetname_data_row['latitude'].iloc[0]
                        crashes.loc[idx, 'longitude'] = long
                        crashes.loc[idx, 'latitude'] = lati


def into_categories(df, column):
    df[column] = df[column].replace(
        ['delivery v', 'delv', 'deliv', 'work van', 'van t', 'van camper'], "van")

    df[column] = df[column].replace(['escooter', 'e-sco', 'e-scoter', 'motor',
                                     'scooter', 'motorbike', 'motorcycle',
                                     'e-scooter', 'motorscooter', 'scoot', ],
                                    "motorbike", 'moped scoo')

    df[column] = df[column].replace(
        ['e bike', 'e-bik', 'ebike', 'e bik', 'e-bike',
         'moped', 'bike', 'minibike', 'minicycle'], "bike")

    df[column] = df[column].replace(
        ['bus', 'taxi', 'schoolbus', 'fire', 'fire truck',
         'school bus', 'commercial', 'pedicab'], "public vehicles")

    df[column] = df[column].replace(
        ['ambul', 'ambulance', 'ambu', 'usps', 'fdny ambul',
         'postal tru', 'us po', 'fdny fire', 'fire engin', 'fdny',
         'firetruck', 'posta'], "service vehicles")

    df[column] = df[column].replace(['truck', 'tanker', 'flat bed',
                                     'rv', 'garbage', 'concrete_mixer',
                                     'concrete mixer', 'tract',
                                     'mack', 'box truck',
                                     'tractor truck diesel',
                                     'dump', 'garbage or refuse', 'carry all',
                                     'tractor truck gasoline',
                                     'tow truck / wrecker', 'chassis cab',
                                     'tanker', 'refrigerated van',
                                     'pick-up truck', 'multi-wheeled vehicle',
                                     'beverage truck', 'amb ', 'armored truck',
                                     'tow t', 'tow truck', 'open body',
                                     'bulk agriculture',
                                     'refg', 'pick up', 'trac', 'garbage tr',
                                     'trk', 'unk', 'box t', 'pick up tr',
                                     'enclosed body - removable enclosure',
                                     'forklift', 'tractor tr'
                                    ], "big_trucks")

    df[column] = df[column].replace(
        ['taxi', 'sedan', 'station wagon/sport utility vehicle',
         'convertible', 'pk', '4 dr sedan', 'passenger vehicle', 'limo', 'sport utility / station wagon',
         '3-door', '2 dr sedan', 'util', 'utili', 'comme', 'motorized home', 'golf cart'], 'car')

    df[column] = df[column].replace(
        ['trailer', 'trail', 'stake or rack', 'flat rack', 'lift boom'], 'vehicle-attachment')
