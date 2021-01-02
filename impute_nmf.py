import pandas as pd
from pandas import datetime
import numpy as np
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)


imputed_nmf = df.copy()   # a copy of DataFrame for storing imputed values

# Hiding values to test imputation
msk = (imputed_nmf.values + np.random.randn(*imputed_nmf.shape) - imputed_nmf.values) < 0.8
imputed_nmf.values[~msk] = 0

# Initializing NMF imputation model
nmf_model = NMF(n_components=5)
nmf_model.fit(imputed_nmf.values)

# Iterative imputation process
while nmf_model.reconstruction_err_**2 > 10:
    W = nmf_model.fit_transform(imputed_nmf.values)
    imputed_nmf.values[~msk] = W.dot(nmf_model.components_)[~msk]
    print(nmf_model.reconstruction_err_)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_nmf[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_nmf.to_csv('./data/AirQualityUCI_NMF.csv', index='Datetime')
