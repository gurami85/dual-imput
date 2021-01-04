import pandas as pd
from pandas import datetime
from sklearn.impute import KNNImputer
import numpy as np
from matplotlib import pyplot as plt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'
input_file = './data/gecco2015_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

arr = df.values

"""
Splitting indices for each data set
    - Air Quality: -1 (impute all data instances)
    - GECCO2015: 483780 (~ 2014-10-20)
"""

split_idx = 483780


"""
Choose the Imputation Method
    - k-NN
    - LOCF
    - NOCB
"""

# Imputation mode: k-NN
imputer = KNNImputer(n_neighbors=2)
imputed = imputer.fit_transform(arr[:split_idx])

# Imputation mode: LOCF
imputed = df.iloc[:split_idx].ffill()
imputed = imputed.fillna(0)
imputed = imputed.values

# Imputation mode: NOCB
imputed = df.iloc[:split_idx].bfill()
imputed = imputed.fillna(0)


"""
Postprocessing after Imputation
"""

# [Option] aggregate train (imputed) / valid (not imputed) data
imputed = np.append(imputed, arr[split_idx:], axis=0)

# Convert to DataFrame
imputed = pd.DataFrame(imputed, index=df.index, columns=df.columns)

# [Option] resampling
imputed = imputed.resample('H').mean()

# [Option] Fill missing values of the resampled data with 0
split_time = df.iloc[[split_idx]].index
# missing values in train set -> fill 0
imputed.loc[
    (imputed.index < split_time.values[0]) &
    (imputed.isnull().any(axis=1) == True)] = 0
# missing values in valid set -> drop
imputed.drop(
    imputed[imputed.isnull().any(axis=1)].index,
    inplace=True)


"""
Visualize and Save
"""

# Visualizing comparison between actual and imputed values
plt.plot(imputed[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed.to_csv('./data/gecco2015_LOCF.csv', index='Datetime')