import pandas as pd
from pandas import datetime

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

imputed_locf = df.ffill()
imputed_locf = imputed_locf.fillna(0)

imputed_nocb = df.bfill()
imputed_nocb = imputed_nocb.fillna(0)

imputed_locf.to_csv('./data/AirQualityUCI_LOCF.csv', index='Datetime')
imputed_nocb.to_csv('./data/AirQualityUCI_NOCB.csv', index='Datetime')
