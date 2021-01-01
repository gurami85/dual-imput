import pandas as pd
from pandas import datetime
import numpy as np

def parser_one(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

df = pd.read_csv('./data/AirQualityUCI.csv',
                 parse_dates=[0],
                 date_parser=parser_one)

df.index = df['Datetime']
df = df.drop(columns=['Datatime'])
df = df.drop(columns=['NMHC(GT)'])  # This feature has many missing values

# Find instances which have missing values
# Then fill all columns with nan (to make completely missing data instances)
df.loc[df.isnull().any(axis=1)] = np.nan

df.to_csv('./data/AirQualityUCI_refined.csv', index='Datetime')

