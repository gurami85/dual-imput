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

# Imputation mode: k-NN
imputer = KNNImputer(n_neighbors=2)
imputed_knn = imputer.fit_transform(arr[:split_idx])

# [Option] aggregate train (imputed) /valid (not imputed) data
imputed_knn = np.append(imputed_knn, arr[split_idx:], axis=0)

# Convert to DataFrame
imputed_knn = pd.DataFrame(imputed_knn, index=df.index, columns=df.columns)

# [Option] resampling
imputed_knn = imputed_knn.resample('H').mean()

# [Option] Fill missing values of the resampled data with 0
# missing values in train set -> fill 0
imputed_knn[imputed_knn.index == '2014-03-30 02:00:00'] = 0
# missing values in valid set -> drop
imputed_knn.drop(
    imputed_knn[imputed_knn.isnull().any(axis=1)].index,
    inplace=True)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_knn[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_knn.to_csv('./data/gecco2015_KNN.csv', index='Datetime')