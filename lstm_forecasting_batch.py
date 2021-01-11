import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt


"""
1. Prepare Feature Set
"""


def parser_one(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def parser_two(x):
    return datetime.strptime(x, '%Y-%m-%d')


input_file = './data/AirQualityUCI_NMF.csv'
input_file = './data/gecco2015_NOCB.csv'
input_file = './data/cnnpred_nasdaq_nocb.csv'


# read the data
df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser_one)

# Strategy 1: fill nan values with 0
# df.fillna(0, inplace=True)

# Strategy 2: drop rows which includes any missing value
# df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

train_ratio = 0.8
split_idx = int(len(df) * 0.8)

split_idx = 8064      # GECCO2015 (hourly)


# Scaling and one-hot encoding
preprocess = make_column_transformer(
    (MinMaxScaler(), df.columns[:]),
    remainder='passthrough')

train = preprocess.fit_transform(df[:split_idx])
valid = preprocess.transform(df[split_idx:])

# Calculate the size of input feature vector
input_size = train.shape[1]-1       # excludes the target variable

# Transformation (ndarray -> torch)
def transform_data(input_data, seq_len):
    x_lst, y_lst = [], []
    size = len(input_data)
    for i in range(size - seq_len):
        # input sequence
        seq = input_data[i:i+seq_len, :input_size]
        # target values of current time steps
        target = input_data[i+seq_len, -1]
        x_lst.append(seq)
        y_lst.append(target)
    x_arr = np.array(x_lst)
    y_arr = np.array(y_lst)
    print("[INFO]x_arr.shape = " + str(x_arr.shape))
    print("[INFO]y_arr.shape = " + str(y_arr.shape))
    return x_arr, y_arr


# Specify a device (gpu|cpu)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float

seq_len = 7
batch_size = 30

x_train, y_train = transform_data(train, seq_len)
x_valid, y_valid = transform_data(valid, seq_len)

# Calculate a number of batches
num_batches = int(x_train.shape[0] / batch_size)

if x_train.shape[0] % batch_size != 0:
    num_batches += 1

"""
2. Model Definition
"""

# Setup the hyperparameters
hidden_size = 30       # default: 32
output_dim = 1
num_layers = 3          # default: 2
learning_rate = 1e-3    # default: 1e-3
num_epochs = 200        # default: 200

# the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = 0
        self.num_layers = num_layers
        # define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    def init_hidden(self):
        # initialize hidden states
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype))
    def forward(self, input):
        # forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input) # [1, batch_size, 24]
        # only take the output from the final time step
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.view(-1)


def weighted_mean_absolute_percentage_error(y_obs, y_hat):
    y_obs, y_hat = np.array(y_obs), np.array(y_hat)
    return np.abs(y_obs - y_hat).sum() / y_obs.sum()


"""
3. Batch Training / Validation
"""

num_exp = 10
hist_exp = np.zeros((num_exp, 2))

for e in range(num_exp):
    model = LSTM(input_size, hidden_size, batch_size=1, output_dim=output_dim, num_layers=num_layers)
    model.seq_len = seq_len
    if torch.cuda.is_available() == True:
        model.cuda()    # for cuda
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    hist = np.zeros((num_epochs, 3))    # train/valid loss history
    for t in range(num_epochs):         # for each epoch
    # for t in range(1):                # [TEST]
        y_pred = np.empty(0)
        for i in range(num_batches):    # for each batch
            print("Training the model: %d/%dth epoch, %d/%dth batch..."
                  % (t + 1, num_epochs, i + 1, num_batches), end='\r')
            # last batch
            if i == num_batches-1:
                x_batch_arr = x_train[i*batch_size:]
                y_batch_arr = y_train[i*batch_size:]
            # other batches
            else:
                x_batch_arr = x_train[i*batch_size:i*batch_size+batch_size]
                y_batch_arr = y_train[i*batch_size:i*batch_size+batch_size]
            # transformation (ndarray -> torch)
            x_batch = Variable(torch.from_numpy(x_batch_arr).float()).type(dtype)
            y_batch = Variable(torch.from_numpy(y_batch_arr).float()).type(dtype)
            model.batch_size = x_batch.shape[0]
            model.hidden = model.init_hidden()
            # get predictions for the batch
            pred_i = model(x_batch)
            # forward pass
            loss_train = loss_fn(pred_i, y_batch)
            # zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()
            # backward pass
            loss_train.backward()
            # update parameters
            optimizer.step()
            # store the predictions
            y_pred = np.append(y_pred, pred_i.detach().cpu().numpy(), axis=0)
        # measure a loss in the current epohch
        loss_train = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_train)).item()
        # Validation step
        x_batch = Variable(torch.from_numpy(x_valid).float()).type(dtype)
        y_forecast = model(x_batch)
        loss_valid = loss_fn(y_forecast, torch.from_numpy(y_valid).type(dtype)).item()
        y_forecast = y_forecast.detach().cpu().numpy()
        loss_wmape = weighted_mean_absolute_percentage_error(y_valid, y_forecast)
        if t == 0:
            loss_train_prev = float('inf')
            loss_valid_prev = float('inf')
        else:
            loss_train_prev = hist[t - 1, 0]
            loss_valid_prev = hist[t - 1, 1]
        print("[INFO] Epoch %d/%d, Train Loss: %.4f, Diff.: %.4f, "
              "Valid Loss: %.4f, Diff.: %.4f"
              % ((t + 1), num_epochs, loss_train, (loss_train - loss_train_prev),
                 loss_valid, (loss_valid - loss_valid_prev)))
        hist[t, 0] = loss_train
        hist[t, 1] = loss_valid
        hist[t, 2] = loss_wmape
    hist_exp[e, 0] = hist[:, 1].min()
    hist_exp[e, 1] = hist[:, 2].min()

mae_max = hist_exp[:, 0].max()
mae_min = hist_exp[:, 0].min()
wmape_max = hist_exp[:, 1].max()
wmape_min = hist_exp[:, 1].min()

print("[INFO] MAE, max: %.4f, min: %.4f, result: %.4f ±%.4f"
      % (mae_max, mae_min, ((mae_max + mae_min)/2), ((mae_max - mae_min)/2) ))

print("[INFO] WMAPE, max: %.4f, min: %.4f, result: %.4f ±%.4f"
      % (wmape_max, wmape_min, ((wmape_max + wmape_min)/2), ((wmape_max - wmape_min)/2) ))



"""
4. Visualization
"""

# default visualization setup
plt.figure(dpi=100)     # set the resolution of plot
# set the default parameters of visualization
color_main = '#2c4b9d'
color_sub = '#00a650'
color_ssub = '#ef9c00'
color_sssub = '#e6551e'
font_family = 'Calibri'
plt.rcParams.update({'font.family': font_family, 'font.size': 23, 'lines.linewidth': 1,
                    "patch.force_edgecolor": True, 'legend.fontsize': 18})


# Training results
plt.plot(y_train, label="Actual")
plt.plot(y_pred, label="Prediction")
plt.legend(loc='best')
plt.show()

# visualize scatter plot
fig, ax = plt.subplots()
ax.scatter(y_valid, y_forecast, 10)   # 10: marker size
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Forecast')
plt.show()

# Validation results
# plt.plot(errors, label="Residual Errors", kind='bar')
plt.plot(y_valid, label="Actual")
plt.plot(y_forecast, label="Forecast")
plt.legend(loc='best')
plt.show()

# visualize scatter plot
fig, ax = plt.subplots()
ax.scatter(y_valid, y_forecast, 10)   # 10: marker size
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Forecast')
plt.show()

