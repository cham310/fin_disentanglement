import os
import numpy as np
import pandas as pd
import pickle as pkl
from time import time
import datetime
from datetime import timedelta
from functools import reduce
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def dataloader(data_directory, batch_size, is_Train):
	dataloader = DataLoader(
		DailyStockPrice(data_directory, train=is_Train),
		batch_size=batch_size, shuffle=True)
	return dataloader

class DailyStockPrice(Dataset):
	"""DailyStockPrice  dataset."""
	def __init__(self, data_dir, train=True, transform = None):
		self.data_dir = data_dir
		self.mode = 'train' if train else 'val'
		self.transform = transform
		self.file_dir = os.path.join(self.data_dir, '{}.pkl'.format(self.mode))
		if not os.path.isfile(self.file_dir):
			self.generate_data()
		self.load_data()

	def generate_data(self, rtn_gap):
		print("No pickle found. Start to generate pickle.")
		# 데이터 로딩해서 수익률 형태로 변환
		# 추후에 원데이터 넣는다면 여기서 코드 바꿔야 함
		self.codes = ['065450','013810','005870','010820','003570','012450', 'ks']
		self.names = ['빅텍', '스페코','휴니드', '퍼스텍', 'S&T중공업', '한화에어로스페이스','KOSPI']
		self.start = '2005-01-01'

		dfList = []
		for code in self.codes:
			df = pd.read_csv(os.path.join(self.data_dir, code + '_daily.csv'), header=0, usecols=[0, 1, 4], names=\
				['Date', 'open', 'close'])
			df['Date_o'] = [datetime.datetime.strptime(str(m), '%Y%m%d') + timedelta(hours=9) for m in df['Date']]
			df['Date_c'] = [datetime.datetime.strptime(str(m), '%Y%m%d') + timedelta(hours=15) for m in df['Date']]
			price_o = pd.DataFrame({'Date': df['Date_o'], code: df['open']})
			price_c = pd.DataFrame({'Date': df['Date_c'], code: df['close']})

			dfList.append(pd.concat([price_o, price_c], axis=0))

		merged = reduce(lambda x, y: pd.merge(x, y, how='outer', on='Date'), dfList)
		merged = merged.set_index('Date')
		merged = merged.sort_index() # 원데이터
		merged = merged[self.start:]  

		rtn = np.log(merged / merged.shift(1)) * 100  # 수익률 계산
# 		rtn['y'] = np.log(merged['ks'] / merged['ks'].shift(rtn_gap)) * 100 # rtn_gap 기간의 kospi 누적 수익률을 tagging
		rtn = rtn.dropna()
		train_data, test_data = train_test_split(rtn.values, test_size=0.1)
		with open(os.path.join(self.data_dir, 'train.pkl'), 'wb') as f:
			pkl.dump(train_data, f, protocol=pkl.HIGHEST_PROTOCOL)
		with open(os.path.join(self.data_dir, 'test.pkl'), 'wb') as f:
			pkl.dump(test_data, f, protocol=pkl.HIGHEST_PROTOCOL)

	def load_data(self):
		with open(self.file_dir, 'rb') as f:
			self.data = pkl.load(f)
		print("Data loaded from {}".format(self.file_dir))
		print(len(self.data))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

if __name__ =='__main__':
	start = time()
	train_dataloader = dataloader(os.path.join(os.getcwd(), 'dataset'), 4, True)
	b = time() - start
	print(b)
	start = time()
	sum_ = 0
	n = 0
	for i in train_dataloader:
		n += 1
		print(i)
		# b = time() - start
		# print(b)
		# sum_ += b
		# start = time()
	print(sum_)
