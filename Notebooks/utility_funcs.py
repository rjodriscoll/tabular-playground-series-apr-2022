import pandas as pd 
import os

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