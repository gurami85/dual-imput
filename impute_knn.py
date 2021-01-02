import pandas as pd
from pandas import datetime
from sklearn.impute import KNNImputer
from matplotlib import pyplot as plt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

# Imputation
imputer = KNNImputer(n_neighbors=2)
imputed_knn = imputer.fit_transform(df)
imputed_knn = pd.DataFrame(imputed_knn, index=df.index, columns=df.columns)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_knn[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_knn.to_csv('./data/AirQualityUCI_KNN.csv', index='Datetime')