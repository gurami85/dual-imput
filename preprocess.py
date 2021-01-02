import pandas as pd
from pandas import datetime
import numpy as np

def parser_one(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

df = pd.read_csv('./data/AirQualityUCI.csv',
                 parse_dates=[0],
                 date_parser=parser_one)

df.index = df['Datetime']
df = df.drop(columns=['Datetime'])
df = df.drop(columns=['NMHC(GT)'])  # This variable includes a lot of missing values
df = df.drop(columns=['T'])         # This variable includes negative values

# Reorder columns of data frame
# [feature1, feature2 ..., target variable]
col_features = df.columns.delete(2)     # features excluded target variable
col_target = df.columns[2]             # target variable: C6H6(GT)

df_arranged = df[col_features]
df_arranged[col_target] = df[col_target]

# # Find instances which have missing values
# # Then fill all columns with nan (to make completely missing data instances)
# df_arranged.loc[df_arranged.isnull().any(axis=1)] = np.nan

# save the arranged data
df_arranged.to_csv('./data/AirQualityUCI_refined.csv', index='Datetime')

