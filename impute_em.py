import pandas as pd
from pandas import datetime
import impyute as impy
from matplotlib import pyplot as plt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

# imputation mode: EM
imputed_em = impy.em(df.values, loops=50)     # default: 50
imputed_em = pd.DataFrame(imputed_em, index=df.index, columns=df.columns)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_em[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_em.to_csv('./data/AirQualityUCI_EM.csv', index='Datetime')

