import pandas as pd
from pandas import datetime
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
import impyute as impy
from matplotlib import pyplot as plt
import missingno


def parser_one(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def parser_two(x):
    return datetime.strptime(x, '%Y-%m-%d')

# parset_one
input_file = './data/AirQualityUCI_refined.csv'
input_file = './data/AirQualityUCI_MICE.csv'
input_file = './data/air_co.csv'
input_file = './data/gecco2015.csv'
input_file = './data/gecco2015-2.csv'
input_file = './data/gecco2015-3.csv'

# parser_two
input_file = './data/gecco2015_refined.csv'
input_file = './data/cnnpred_nasdaq_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser_one)

imputed_mice = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser_one)

"""
Splitting indices for each data set
    - Air Quality: entire (impute all data instances)
    - Air Quality (CO): 7494(case deletion), entire (~ 2005-01-16)
    - GECCO 2015 (Hourly): 483780 (~ 2014-10-20)
    - GECCO 2015-2: 264900 (~ 2014-5-21)
        + No missing values of target variable
    - GECCO 2015-3: 211680 (case deletion), entire (2014-05-22 ~ 2014-11-21)
    - CNN Pred: len(df)
"""


# Case deletion: remove cases in test set whose target values are missing
#   - Air Quality (CO)
#   - GECCO 2015-3

split_idx = 7494
split_time = df.iloc[[split_idx]].index

del_cases = df.loc[ (df.index >= split_time.values[0]) &\
        (df['Return_Temperature'].isnull() == True) ].index

df.drop(del_cases, inplace=True)

# Splitting index and time for determining an imputation scope
split_idx = len(df)

# Case: Impute only values in the training set
split_time = df.iloc[[split_idx]].index

"""
Choose the Imputation Method
    - MICE
    - k-NN
    - EM
    - LOCF
    - NOCB
    - Case Deletion
    - Zero Substitution
"""

# [Imputation mode: NMF]
imputed = df.copy().values[:split_idx]

#   hiding values to test imputation
msk = (imputed + np.random.randn(*imputed.shape) - imputed) < 0.8
imputed[~msk] = 0

#   initializing NMF imputation model
nmf_model = NMF()     # n_components: num. of features
nmf_model.fit(imputed)

#   iterative imputation process
# while nmf_model.reconstruction_err_**2 > 10:
while nmf_model.reconstruction_err_ > 2.5:
    W = nmf_model.fit_transform(imputed)
    imputed[~msk] = W.dot(nmf_model.components_)[~msk]
    print(nmf_model.reconstruction_err_)

# [Imputation mode: MICE]
imputed = impy.mice(df.values[:split_idx])

# [Imputation mode: k-NN]
imputer = KNNImputer(n_neighbors=10)    # default: 2
imputed = imputer.fit_transform(df.values[:split_idx])

# [Imputation mode: EM]
imputed = impy.em(df.values[:split_idx], loops=50)

# [Imputation mode: LOCF]
imputed = df.copy().iloc[:split_idx].ffill()
imputed = imputed.fillna(0)
imputed = imputed.values

# [Imputation mode: NOCB]
imputed = df.copy().iloc[:split_idx].bfill()
imputed = imputed.fillna(0)
imputed = imputed.values

# [No imputation: Case Deletion]
imputed = df.drop(df[df.isnull().any(axis=1)].index).copy()

# [No imputation: Zero Substitution]
imputed = df.copy().iloc[:split_idx].fillna(0)

# [No imputation: Mean Substitution]
imputed = df.copy().iloc[:split_idx].fillna(df.mean())


"""
Postprocessing after Imputation
"""

# [Option] aggregate train (imputed) / valid (not imputed) data
imputed = np.append(imputed, df.values[split_idx:], axis=0)

# Convert to DataFrame
#imputed = pd.DataFrame(imputed, index=df.index, columns=df.columns)
imputed = pd.DataFrame(imputed, index=df.iloc[:split_idx].index, columns=df.columns)

# [Option] resampling
imputed = imputed.resample('H').mean()

# [Option] Fill missing values of the resampled data with 0

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

# # [TEMP]
# # Visualizing comparison between actual and imputed values
#
# plt.plot(imputed_locf[df.columns[-1]], label='LOCF')
# plt.plot(imputed_nocb[df.columns[-1]], label='NOCB')
# plt.plot(imputed_mean[df.columns[-1]], label='Mean substitution')
# plt.plot(imputed_em[df.columns[-1]], label='EM')
# plt.plot(imputed_mice[df.columns[-1]], label='MICE')
# plt.plot(imputed_knn[df.columns[-1]], label='k-NN')
# plt.plot(df[df.columns[-1]], label='Actual')
# plt.legend(loc='best')
# plt.show()
#
# imputed_knn.to_csv('./data/air_co_knn_temp.csv', index='Datetime')

# Visualizing a nullity matrix using missingno
missingno.matrix(df,figsize=(10,5), fontsize=12);

# Save the data set with imputed values

imputed.to_csv('./data/AirQualityUCI_MEAN.csv', index='Datetime')
imputed.to_csv('./data/gecco2015-2_mean.csv', index='Datetime')
imputed.to_csv('./data/gecco2015-3_mean.csv', index='Datetime')
imputed.to_csv('./data/cnnpred_nasdaq_mean.csv', index='Datetime')
