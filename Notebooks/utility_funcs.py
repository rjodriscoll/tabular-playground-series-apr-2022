import pandas as pd 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_train_labels_test():
    is_kaggle = False
    data_path = '../Data/'
    if not os.path.exists(data_path):
        is_kaggle = True
        data_path = '/kaggle/input/"tabular-playground-series-apr-2022/'

    train = pd.read_csv(data_path + "train.csv")
    labels = pd.read_csv(data_path + "train_labels.csv")
    test = pd.read_csv(data_path + "test.csv")

    return train, labels, test


def split_train_data(train, labels, train_perc = 0.9):
    length = len(train)
    train_size = int(length * train_perc) - int((length * train_perc) % 60)
    length_y = len(labels)
    train_size_y = int(length_y * train_perc)
    X_train, X_test = train[0:train_size], train[train_size:length]
    y_train, y_test = labels['state'][0:train_size_y], labels['state'][train_size_y:length_y]

    return X_train, X_test, y_train, y_test

def scale_and_as_array(train, features, test, X_train, X_test,y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(train[features])
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    test[features] = scaler.transform(test[features])
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return test, X_train, X_test, y_train, y_test 
