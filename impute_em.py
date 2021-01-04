import pandas as pd
from pandas import datetime
import impyute as impy
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
    - GECCO2015: 154140
"""

split_idx = 154140

# Imputation mode: EM
imputed_em = impy.em(arr[:split_idx], loops=50)     # default: 50

# [Option] aggregate train (imputed) /valid (not imputed) data
imputed_em = np.append(imputed_em, arr[split_idx:], axis=0)

# [Option] resampling
imputed_em = imputed_em.resample('D').mean()

# Convert to DataFrame
imputed_em = pd.DataFrame(imputed_em, index=df.index, columns=df.columns)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_em[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_em.to_csv('./data/gecco2015_EM.csv', index='Datetime')

