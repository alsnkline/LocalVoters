"""
Provides various utils.

Load clean households, voter data.
Find_changes between two dataframes
"""

# imports
import pandas as pd
import numpy as np


def load_households(file):
    """Loading household data from provided file."""
    df = pd.read_csv(file)
    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    # print(df.info())
    return df


def load_voters(file):
    """Loading voter data from provided file."""
    df = pd.read_csv(file)
    df.RegDate = pd.to_datetime(df.RegDate)
    df.RegDateOriginal = pd.to_datetime(df.RegDateOriginal)
    cols_for_fill = ['E6_110816', 'E5_060716', 'E4_110414', 'E3_060314', 'E2_110612', 'E1_060512', 'E5_060716BT', 'E1_060512BT']
    df[cols_for_fill] = df[cols_for_fill].apply(lambda x: x.fillna(''))
    cols_for_cat = ['E6_110816', 'E5_060716', 'E4_110414', 'E3_060314', 'E2_110612', 'E1_060512',
                    'MailCountry', 'StreetType', 'E5_060716BT', 'E1_060512BT', 'PAV', 'Party',
                    'BirthPlaceState', 'BirthPlaceCountry', 'Gender']
    df[cols_for_cat] = df[cols_for_cat].apply(lambda x: x.astype('category'))
    # print(df.info())
    return df


def find_changes(df1, df2):
    """Finding diffs in the data from first dump to second dump."""
    # https://stackoverflow.com/a/17095620/1215012

    ne_stacked = (df1 != df2).stack()
    changed = ne_stacked[ne_stacked]
    changed.index.names = ['id', 'col']

    difference_locations = np.where(df1 != df2)
    changed_from = df1.values[difference_locations]
    changed_to = df2.values[difference_locations]

    out = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)
    out = out.reset_index()
    interesting = out.loc[(out['from'].notnull()) & (out['to'].notnull())]
    print(interesting.shape)
    print(interesting.col.unique())
    return interesting
