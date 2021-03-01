import pandas as pd
import numpy as np
import os.path as pth
from sklearn.preprocessing import MinMaxScaler
from time import sleep

np.set_printoptions(suppress=True, threshold=50000)  # to see full output


FILE_NAME = './SEMA_2013_2016_hourly.csv'
OUTPUT_FILE_NAME = './SEMA_2013_2016_hourly_cleaned.csv'


def clean_form(x):
    """
    if x is of form: 1210.2  make no change
    else if x is of form '1,210.2': change it into 1210.2
    """
    x = str(x)
    if "," not in x:
        return np.double(x)
    else:
        x = ''.join(x.split(','))
        return np.double(x)


def clean_data():
    df = pd.read_csv(FILE_NAME, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    df['DA_DEMD'] = df['DA_DEMD'].apply(lambda x: clean_form(x))
    return df
    

def Power_data_load(cleaned_data, sequence_length=24):
    columns = ['DA_DEMD', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC', 'RT_LMP',
               'RT_EC', 'RT_CC', 'RT_MLC', 'DryBulb', 'DewPnt',  'DEMAND']
    df = cleaned_data[columns]
    # print(df)
    data_all = np.array(df).astype(float)
    print('data_all.shape" {}'.format(data_all.shape))
    feature_data_all = data_all[:, :-1]
    output_data_all = data_all[:, -1].reshape((-1, 1))
    feature_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    feature_data_all = feature_scaler.fit_transform(feature_data_all)
    output_data_all = output_scaler.fit_transform(output_data_all)
    data = []
    for i in range(len(feature_data_all) - sequence_length):
        data.append(feature_data_all[i: i + sequence_length])
    reshaped_data = np.array(data).astype('float64')
    # np.random.shuffle(reshaped_data)
    # uniform x but not y
    # print('reshaped_data.shape: {}'.format(reshaped_data.shape))
    #
    data_x = reshaped_data   # select all elements except the last one
    data_y = output_data_all # .reshape((-1, 24, 1))    # select the last element
    # print('data_x: {}'.format(data_x[:10]))
    # print('data_y: {}'.format(data_y[:10]))
    # print('data_y[0]: {}'.format(data_y[0]))

    DATASET_SIZE = int(reshaped_data.shape[0])
    print('data_x: {}; data_y: {}'.format(data_x.shape, data_y.shape))
    # data_x: (35040, 24, 11) data_y: (35040, 1)
    return data_x, data_y, DATASET_SIZE, feature_scaler, output_scaler


if __name__ == '__main__':
    cleaned_data = clean_data()
    # data_x, data_y, DATASET_SIZE, _ = Power_data_load(cleaned_data)
    data_x, data_y, DATASET_SIZE, feature_scaler, output_scaler = Power_data_load(cleaned_data)
    print("Check if there is any Data is NAN:", np.argwhere(np.isnan(data_x)))
    print("Check if there is any Data is NAN:", np.argwhere(np.isnan(data_y)))
    # print(data_x)
    # print('-'*82)
    # print(data_y)
