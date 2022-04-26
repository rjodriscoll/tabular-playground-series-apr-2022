import pandas as pd 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_train_labels_test(is_py = False):
    data_path = 'Data/' if is_py else '../Data/'   

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
    y_train, y_test = labels[0:train_size_y], labels[train_size_y:length_y]

    return X_train, X_test, y_train, y_test

def scale_and_as_array(data, features, scaler, scale_data = False):
    data = data.copy() # this is to prevent pandas copy warning
    if scale_data:
        data[features] = scaler.transform(data[features])
    data_prepped = np.array(data[features])
    return data_prepped
