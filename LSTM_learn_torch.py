"""
Terry 2021/02/27
fun: LSTM for data forecasting
"""
import os
import json
import argparse
import random
import os.path as pth
from time import sleep
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from ReadSEMA import *
matplotlib.use('Agg')

# add args
parser = argparse.ArgumentParser(description='parameters for LSTM Training')
parser.add_argument('--patience', default=10, type=int, help='how long to wait after last time validation loss improved')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--epoches', default=20, type=int, help='maximum training epoches')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--dropout', default=0.2, type=float, help='drop out rate')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--linear_hidden_size1', default=64, type=int, help='hidden size for fuly connected layer')
parser.add_argument('--bidirection', default=False, type=bool, help='bidirection in LSTM: True/False')
parser.add_argument('--feature_size', default=11, type=int, help = 'feature size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
FILE_NAME = "./SEMA_2013_2016_hourly.csv"
SMALL_NUM = 0.00001

def init_weights(m):
    # if type(m) == nn.Linear:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)


class Net(nn.Module):
    """encode user features include description and other counts
    text_input_size: text embedding size
    user_feat_size: dim of input list of [counts]
    """
    def __init__(self, input_size=args.feature_size, seq_len=24, num_layer=4, hidden_size=50, dropout=0.2):
        super(Net, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = args.batch_size
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.num_layer, dropout=self.dropout)
        self.Linear1 = nn.Linear(self.hidden_size * self.seq_len, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, 1)
        # self.linear.apply(init_weights)

    def forward(self, input):
        """
        :param input: of shape `(seq_len, batch, input_size)`
        :param h0: of shape `(num_layers * num_directions, batch, hidden_size)`
        :param c0:  of shape `(num_layers * num_directions, batch, hidden_size)`
        :return: output, (h_n, c_n)
        notice: seq_len is so-called 'time_step'

        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.
        - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        """
        h0_size = [self.num_layer, self.batch_size, self.hidden_size]
        c0_size = [self.num_layer, self.batch_size, self.hidden_size]
        h0 = torch.randn(h0_size).to(device)
        c0 = torch.randn(c0_size).to(device)
        input = input.permute(1, 0, 2)  # change the order to '(seq * batch * len)'
        # print("input.shape: ", input.shape) # torch.Size([24, 64, 11])

        # seq_len=24, num_layer=4, hidden_size=50, batch_size=64
        # output1, (_, _) = self.LSTM(input, h0, c0)  # batch * dim
        output, (hn, cn) = self.LSTM(input, (h0, c0))  # batch * dim
        # print("output1.shape: ", output1.shape)   # torch.Size([24, 64, 50])
        # print("hn.shape: ", hn.shape)   # torch.Size([4, 64, 50])
        # print("cn.shape: ", cn.shape)   # torch.Size([4, 64, 50])
        # output1 = output[-1, :, :]
        output = output.permute(1, 0, 2)
        output = output.contiguous()
        batch_size = output.size(0)
        output1 = output.view(batch_size, -1)
        # print('output1.shape: ', output1.shape)  # torch.Size([64, 50])
        output1 = self.Linear1(output1)
        output2 = F.relu(output1)
        # print('output2.shape: ', output2.shape)
        output3 = self.Linear2(output2)
        # print('output3.shape: ', output3.shape)  # torch.Size([64, 1])
        return output3


class SEMAData(Dataset):
    """
    getitem: return data
    """

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        # data_x: (35040, 24, 11) data_y: (35040, 24)
        data_x = torch.Tensor(self.data_x[index])
        data_y = torch.Tensor(self.data_y[index])
        # print("data_x.shape", data_x.shape)
        # print("data_y.shape", data_y.shape)
        # print('----------')

        return data_x, data_y  # data_x: 24*11, data_y: 24


def get_dataloader(data_x, data_y, DATASET_SIZE, seed=1):
    """
    :param batch_size: batch_size
    :param seed: random seed
    :param DATASET_SIZE: DATASET_Size
    :return: train_loader, test_loader
    """
    # data_x, data_y, DATASET_SIZE, _ = data_load(FILE_NAME, sequence_length=24)
    LoadData = SEMAData(data_x, data_y)

    indices = list(range(DATASET_SIZE))
    random.seed(seed)
    random.shuffle(indices)
    split = int(DATASET_SIZE * 0.8)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_data = DataLoader(LoadData, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    test_data = DataLoader(LoadData, batch_size=args.batch_size, sampler=test_sampler, drop_last=True)

    return train_data, test_data


def adjust_learning_rate(optim, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def _compute_accuracy(y_pred, y_labels):
    # return 1.0*(sum(abs(y_pred-y_labels)/abs(y_labels)))
    # return 1.0*torch.mean(torch.true_divide(abs(y_pred - y_labels), y_labels))
    acc = 1.0*torch.mean(torch.true_divide(abs(y_pred-y_labels), abs(y_labels)))
    return acc


def _train_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)

    model.train()
    global_step = 0
    for epoch in range(args.epoches):
        batch_train_err = []
        batch_train_loss = []
        batch_counter = 0
        for data in train_data:
            data_x, data_y = data[0].to(device), data[1].to(device)
            # print("data_x.shape: ", data_x.shape)
            # print("data_y.shape: ", data_y.shape)
            # adjust_learning_rate(optimizer, epoch)

            optimizer.zero_grad()
            # print("----------------start feeding into model------------")
            output = model(data_x)
            batch_counter += 1
            global_step += 1
            # print(output)
            # print("data_y shape: ", data_y.shape)
            # print("data_y : ", data_y)
            # print("output shape: ", output.shape)
            # print("output : ", output)
            loss = loss_fun(output, data_y)
            err = _compute_accuracy(output, data_y)
            loss.backward()
            optimizer.step()

            batch_train_err.append(err.item())
            batch_train_loss.append(loss.item())

            if global_step % 20 == 0:
                print("epoch: {} | batch: {:03d} | loss: {:.5f} | accuracy: {:.5f}" \
                      .format(epoch + 1, batch_counter, loss.item(), 1.0-err.item()))
                # fw_log.write("epoch: {} global step: {} loss: {:.5f} train accuracy: {:.5f}\n" \
                #              .format(epoch + 1, global_step, loss.item(), accy))
        # print('batch_train_err', batch_train_err)
        # print('batch_train_loss', batch_train_loss)
        #
        # batch_train_err = torch.FloatTensor(batch_train_err).reshape(-1, 1)x
        # batch_train_loss = torch.FloatTensor(batch_train_loss).reshape(-1, 1)
        # print(batch_train_err)
        # print(batch_train_loss)


        # batch_accy = 1.0 - torch.mean(batch_train_err, dim=0).item()
        # batch_loss = torch.mean(batch_train_loss, dim=0).item()

        # print(batch_accy)
        # print(batch_loss)
        # print('----------------*************------------------')
        # print("Training accuracy: {:2f} | Training loss: {:2f}\n".format(batch_accy*100, batch_loss))
    print("Training finished!")


def _test_model():
    test_accy = []
    all_output = []
    all_true_y = []
    model.eval()

    with torch.no_grad():
        for data in test_data:
            data_x, data_y = data[0].to(device), data[1].to(device)
            output = model(data_x)
            all_true_y.append(data_y)
            all_output.append(output)
            accy = _compute_accuracy(output, data_y)
            test_accy.append(accy)
        print("Testing percentage error: {:2f}\n".format(1.0-sum(test_accy)/len(test_accy)))
    print("Training finished!")
    all_output = torch.cat(all_output, dim=0)
    true_y = torch.cat(all_true_y, dim=0)
    print(all_output.shape)
    return all_output, true_y


if __name__ == '__main__':

    # FILE_NAME = "./SEMA_2013_2016_hourly.csv"

    cleaned_data = clean_data()
    data_x, data_y, DATASET_SIZE, feature_scaler, output_scaler= Power_data_load(cleaned_data)
    loss_fun = nn.MSELoss()

    model = Net()  # load model with default parameters
    model = model.to(device)
    # data_x: (35040, 24, 11) data_y: (35040, 1)
    # print("Check if there is any Data is NAN:", np.argwhere(np.isnan(data_x)))
    # print("Check if there is any Data is NAN:", np.argwhere(np.isnan(data_y)))
    # sleep(100)

    train_data, test_data = get_dataloader(data_x, data_y, DATASET_SIZE, seed=1)
    _train_model()
    predict_y, true_y = _test_model()

    true_y = true_y.detach().cpu().clone().numpy()
    predict_y = predict_y.detach().cpu().clone().numpy()
    print(true_y.shape)
    print(predict_y.shape)
    print('*'*88)
    predict_y = output_scaler.inverse_transform(predict_y)
    true_y = output_scaler.inverse_transform(true_y)
    print(true_y.shape)
    print(predict_y.shape)
    print('*'*88)
    # predict_y = scaler.inverse_transform([[i] for i in predict_y])
    # test_y = scaler.inverse_transform([[i] for i in true_y])
    fig2 = plt.figure(2)
    plt.plot(predict_y[:200], 'r:')
    plt.plot(true_y[:200], 'g-')
    plt.legend(['predict', 'true'])
    plt.show()
    plt.savefig('figure/1.png')

