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

# imputation mode: MICE
imputed_mice = impy.mice(df.values)
imputed_mice = pd.DataFrame(imputed_mice, index=df.index, columns=df.columns)

# Visualizing comparison between actual and imputed values
plt.plot(imputed_mice[df.columns[0]], label='imputed')
plt.plot(df[df.columns[0]], label='actual')
plt.legend(loc='best')
plt.show()

# Save the data set with imputed values
imputed_mice.to_csv('./data/AirQualityUCI_MICE.csv', index='Datetime')

