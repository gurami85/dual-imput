import pandas as pd
from pandas import datetime


def parser_one(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

def parser_two(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def parser_three(x):
    return datetime.strptime(x, '%Y-%m-%d')

"""
Dataset 1: Air Quality
"""

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


"""
Dataset 2: Heating System (GECCO2015 Challenge)
    - Selected data: 2013-11-19 ~ 2015-01-12, 604,800 (df[1128:605928])
    - Target variable: Return temperature
"""

df = pd.read_csv('./data/gecco2015.csv',
                 parse_dates=[0],
                 date_parser=parser_two)

df.index = df['Timestamp']
df.drop(columns=['Timestamp'], inplace=True)

# Select a period of dataset
df_sel = df.iloc[1128:605928].copy()

col_features = df.columns.delete(2)
col_target = df.columns[2]

df_arranged = df_sel[col_features]
df_arranged[col_target] = df_sel[col_target]

df_arranged.to_csv('./data/gecco2015_refined.csv', index='Datetime')

"""
Dataset: CNNPred (NASDAQ)
"""

df = pd.read_csv('./data/cnnpred_nasdaq.csv',
                 parse_dates=[0],
                 date_parser=parser_three)

df.index = df['Datetime']
df.drop(columns=['Datetime'], inplace=True)

col_features = df.columns.delete(0)
col_target = df.columns[0]

df_arranged = df[col_features]
df_arranged[col_target] = df[col_target]

df_arranged.to_csv('./data/cnnpred_nasdaq_refined.csv', index='Datetime')