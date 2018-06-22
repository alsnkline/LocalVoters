"""
Provides various utils:
Load clean households and voter data
"""

# imports
import pandas as pd

def load_households(file):
    df = pd.read_csv(file)
    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    #print(df.info())
    return df

def load_voters(file):
    df = pd.read_csv(file)
    df.RegDate = pd.to_datetime(df.RegDate)
    df.RegDateOriginal = pd.to_datetime(df.RegDateOriginal)
    cols_for_fill = ['E6_110816','E5_060716','E4_110414','E3_060314','E2_110612','E1_060512','E5_060716BT','E1_060512BT']
    df[cols_for_fill] = df[cols_for_fill].apply(lambda x: x.fillna(''))
    cols_for_cat = ['E6_110816','E5_060716','E4_110414','E3_060314','E2_110612','E1_060512',
                'MailCountry','StreetType','E5_060716BT','E1_060512BT','PAV','Party',
               'BirthPlaceState','BirthPlaceCountry','Gender']
    df[cols_for_cat] = df[cols_for_cat].apply(lambda x: x.astype('category'))
    #print(df.info())
    return df