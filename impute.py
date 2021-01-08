import pandas as pd
from pandas import datetime
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
import impyute as impy
from matplotlib import pyplot as plt

def parser_one(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def parser_two(x):
    return datetime.strptime(x, '%Y-%m-%d')

input_file = './data/AirQualityUCI_refined.csv'
input_file = './data/gecco2015_refined.csv'
input_file = './data/cnnpred_nasdaq_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser_two)


"""
Splitting indices for each data set
    - Air Quality: len(df) (impute all data instances)
    - GECCO 2015: 483780 (~ 2014-10-20)
    - CNN Pred: len(df)
"""

split_idx = len(df)


"""
Choose the Imputation Method
    - k-NN
    - LOCF
    - NOCB
"""

# Imputation mode: NMF
imputed = df.copy().values[:split_idx]

#   hiding values to test imputation
msk = (imputed + np.random.randn(*imputed.shape) - imputed) < 0.8
imputed[~msk] = 0

#   initializing NMF imputation model
nmf_model = NMF(n_components=4)     # n_components: num. of features
nmf_model.fit(imputed)

#   iterative imputation process
# while nmf_model.reconstruction_err_**2 > 10:
while nmf_model.reconstruction_err_ > 2.45:
    W = nmf_model.fit_transform(imputed)
    imputed[~msk] = W.dot(nmf_model.components_)[~msk]
    print(nmf_model.reconstruction_err_)

# Imputation mode: MICE
imputed = impy.mice(df.values[:split_idx])

# Imputation mode: k-NN
imputer = KNNImputer(n_neighbors=2)
imputed = imputer.fit_transform(df.values[:split_idx])

# Imputation mode: LOCF
imputed = df.copy().iloc[:split_idx].ffill()
imputed = imputed.fillna(0)
imputed = imputed.values

# Imputation mode: NOCB
imputed = df.copy().iloc[:split_idx].bfill()
imputed = imputed.fillna(0)
imputed = imputed.values

# No imputation: Case Deletion
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
df.to_csv('./data/cnnpred_nasdaq_deleted.csv', index='Datetime')


"""
Postprocessing after Imputation
"""

# [Option] aggregate train (imputed) / valid (not imputed) data
imputed = np.append(imputed, df.values[split_idx:], axis=0)

# Convert to DataFrame
imputed = pd.DataFrame(imputed, index=df.index, columns=df.columns)

# [Option] resampling
imputed = imputed.resample('H').mean()

# [Option] Fill missing values of the resampled data with 0
split_time = df.iloc[[split_idx]].index
#   missing values in train set -> fill 0
imputed.loc[
    (imputed.index < split_time.values[0]) &
    (imputed.isnull().any(axis=1) == True)] = 0
#   missing values in valid set -> drop
imputed.drop(
    imputed[imputed.isnull().any(axis=1)].index,
    inplace=True)


"""
Visualize and Save
"""

# Visualizing comparison between actual and imputed values
plt.plot(imputed[df.columns[-1]], label='imputed')
plt.plot(df[df.columns[-1]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed.to_csv('./data/cnnpred_nasdaq_knn.csv', index='Datetime')