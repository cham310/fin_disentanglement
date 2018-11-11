import os
import numpy as np
import pandas as pd
import pickle as pkl
from time import time
from datetime import datetime
from datetime import timedelta
from functools import reduce
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from configuration import get_config
from bisect import bisect
from configuration import get_config


def dataloader(data_directory, batch_size, is_Train):
    dataloader = DataLoader(
        DailyStockPrice(data_directory, train=is_Train),
        batch_size=batch_size, shuffle=False)
    return dataloader

class DailyStockPrice(Dataset):

    """DailyStockPrice  dataset."""

    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.mode = 'train' if train else 'test'
        self.transform = transform
        self.file_dir = os.path.join(self.data_dir, '{}.pkl'.format(self.mode))
        if not os.path.isfile(self.file_dir):
            self.generate_data()
        self.load_data()

    def generate_data(self):
        print("No pickle found. Start to generate pickle.")

        # 방산주 6종목 + KOSPI 대상
        self.codes = ['065450', '013810', '005870', '010820', '003570', '012450', 'ks']
        self.names = ['빅텍', '스페코', '휴니드', '퍼스텍', 'S&T중공업', '한화에어로스페이스', 'KOSPI']
        start = '2004-01-01'

        dfList = []
        for code in self.codes:
            df = pd.read_csv(os.path.join(self.data_dir, code + '_daily.csv'), header=0, usecols=[0, 1, 4], names= \
                ['Date', 'open', 'close'])
            df['Date_o'] = [datetime.strptime(str(m), '%Y%m%d') + timedelta(hours=9) for m in df['Date']]
            df['Date_c'] = [datetime.strptime(str(m), '%Y%m%d') + timedelta(hours=15) for m in df['Date']]
            price_o = pd.DataFrame({'Date': df['Date_o'], code: df['open']})
            price_c = pd.DataFrame({'Date': df['Date_c'], code: df['close']})

            dfList.append(pd.concat([price_o, price_c], axis=0))

        ##merged: 원데이터, rtn: 로그 수익률
        merged = reduce(lambda x, y: pd.merge(x, y, how='outer', on='Date'), dfList)
        merged = merged.set_index('Date')
        merged = merged.sort_index()

        rtn = np.log(merged / merged.shift(1)) * 100  # 수익률 계산
        rtn = rtn[start:]
        rtn = rtn.dropna()

        temp1, temp2 = train_test_split(rtn.values, test_size=0.1, random_state=0, shuffle = False)
        self.date = rtn.index

        # train_set
        self.trainset = []
        self.train_data = []
        self.train_label = []
        for i in range(temp1.shape[0] - 10 + 1):
            self.train_data.append(np.asarray(temp1[i:(i + 10), :-1]))
            self.train_label.append(np.sum(np.asarray(temp1[i:(i + 10), -1])))
        self.trainset = (self.train_data, self.train_label)

        # test_set
        self.testset = []
        self.test_data = []
        self.test_label = []
        for i in range(temp2.shape[0] - 10 + 1):
            self.test_data.append(np.asarray(temp2[i:(i + 10), :-1]))
            self.test_label.append(np.sum(np.asarray(temp2[i:(i + 10), -1])))
        self.testset = (self.test_data, self.test_label)

        with open(os.path.join(self.data_dir, 'train.pkl'), 'wb') as f:
            pkl.dump(self.trainset, f, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(self.data_dir, 'test.pkl'), 'wb') as f:
            pkl.dump(self.testset, f, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(self.data_dir, 'date.pkl'), 'wb') as f:
            pkl.dump(self.date, f, protocol=pkl.HIGHEST_PROTOCOL)

    def load_data(self):
        with open(self.file_dir, 'rb') as f:
            self.data, self.label = pkl.load(f)
        self.interval = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        # self.interval = [-3, -1, 0, 1, 3]

        print("Data loaded from {}".format(self.file_dir))
        print("num of trainset:{}".format(len(self.label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = bisect(self.interval, self.label[idx])
        return self.data[idx], label

    def get_date(self, idx, batch_size, mode): # mode가 true이면 train
        with open(os.path.join(self.data_dir, 'date.pkl'), 'rb') as f:
            date = pkl.load(f)
            if mode == True:
                start = 0
            else: start = int(len(date)*0.9)-9
            out = date[start+idx:start+idx+batch_size]
        return out

if __name__ == '__main__':
    start = time()
    train_dataloader = dataloader(os.path.join(os.getcwd(), 'dataset'), 4, True)
    b = time() - start
    start = time()
    sum_ = 0
    n = 0
    for data, label in train_dataloader:
    #     n += 1
        print(data.size())
        print(label)
        break
    # b = time() - start
    # print(b)
    # sum_ += b

