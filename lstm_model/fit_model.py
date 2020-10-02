from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from model import get_model
import argparse
import os

parser = argparse.ArgumentParser(
    description="Using LSTM to predict the pollution of china")

parser.add_argument("-path_dataset", action="store", required=True,
                    help="The dataset of the pollution china", dest="path_dataset")

parser.add_argument("-experiment_name", action="store", required=True,
                    help="Folder to save the experiment", dest="experiment_name")

parser.add_argument("-variables_name", action="store", required=True,
                    help="Variable for prediction the movement", dest="variables_name")

arguments = parser.parse_args()

path_dataset = arguments.path_dataset
experiment_name = arguments.experiment_name
variables_name = arguments.variables_name


# load the dataset
df = pd.read_csv(path_dataset, header=0, index_col=0)

# Preprocessing

data_split = int(df.shape[0]*0.8)

variables = {}
for i, col in enumerate(df.columns):
    variables[col] = i

if 0 < variables[variables_name] < df.shape[1]:
    data_set = df.iloc[:, variables[variables_name]: variables[variables_name]+1].values

training_set = data_set[:data_split, :]
test_set = data_set[data_split:, :]


sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)
training_set_scaled = data_set_scaled[:data_split]
test_set_scaled = data_set_scaled[data_split:]

X_train = []
y_train = []

for i in range(30, training_set.shape[0]):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
y_test = []

for i in range(30, test_set.shape[0]):
    X_test.append(test_set_scaled[i-30:i, 0])
    y_test.append(test_set_scaled[i, 0])


X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model = get_model(X_train)

model.compile(loss='mse', optimizer='Adam')


model.fit(X_train[:X_train.shape[0]*80//100], y_train[:X_train.shape[0]*80//100], epochs=100, batch_size=512,
          validation_data=(X_train[X_train.shape[0]*80//100:], y_train[X_train.shape[0]*80//100:]), verbose=2, shuffle=False)

model.save(os.path.join(experiment_name, "model.h5"))

