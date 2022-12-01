import numpy as np
import pandas as pd
from multi_layered_perceptron import *

def main():
  # load training data
  df_train = pd.read_csv("data/features_30_sec.csv")
  df_train['label'] = df_train['label'].replace(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  # shuffle the data
  df_train = shuffle_rows(df_train)

  # split train and validation set
  train_val_split = 0.8
  train_size = round(df_train.shape[0] * train_val_split)
  data_train = df_train[:train_size,:].T
  data_val = df_train[train_size:,:].T
  
  # divide input features and target feature
  X_train = data_train[1:-1]
  y_train = data_train[-1]
  X_val = data_val[1:-1]
  y_val = data_val[-1]
  
  # normalize training and val sets
  X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
  X_val = (X_val-X_val.min())/(X_val.max()-X_val.min())

  # set network and optimizer parameters  
  layers_dims = [58, 10, 10, 10, 10]
  max_iter = 1000
  alpha = 0.1

  # train the network
  params = gradient_descent_optimization(X_train, y_train, layers_dims, max_iter, alpha)
  
if __name__ == '__main__':
  main()
