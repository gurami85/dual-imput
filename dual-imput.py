import pandas as pd
from pandas import datetime
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from autoimpute.imputations import SingleImputer, MultipleImputer
from matplotlib import pyplot as plt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

selection_ratio = 0.25
n_selection = int(df.shape[1] * selection_ratio)

df_nanless = df.loc[df.isnull().any(axis=1) == False]
x_nanless = df_nanless.values[:, :-1]
y_nanless = df_nanless.values[:, -1]

selector = SelectKBest(score_func=f_regression, k=n_selection)
selector.fit_transform(x_nanless, y_nanless)
scores = selector.scores_
idx_selected = [sorted(scores, reverse=True).index(x) for x in scores][:n_selection]

"""
Imputation (1st Round): Univariate Imputation
    - Imputation method: Quadratic spline interpolation
    - Impute selected k-features
"""

strategy = "interpolate"
fill_strategy = "cubic"

dict_strategy = dict()
dict_imp_kwgs = dict()

for i in idx_selected:
    dict_strategy.update({df.columns[i]: strategy})
    dict_imp_kwgs.update({df.columns[i]: {'fill_strategy': fill_strategy}})

imp_x = SingleImputer(
    strategy=dict_strategy,
    imp_kwgs=dict_imp_kwgs
)

df_imputed = imp_x.fit_transform(df)

plt.plot(df_imputed[df.columns[idx_selected[0]]], label='Imputed')
plt.plot(df[df.columns[idx_selected[0]]], label='Actual')

train_ratio = 0.8
split_idx = int(len(df) * train_ratio)

x_train = df[:split_idx].values[1:]
y_train = df[:split_idx].values[0]
x_test = df[split_idx:].values[1:]
y_test = df[split_idx:].values[0]




