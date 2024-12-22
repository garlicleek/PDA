import wfdb
import time
import random

from src.pdd_parser import *
from src.utils import *

import sys
import logging

# 配置日志
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(message)s',
	handlers=[
		logging.FileHandler("training.log"),  # 保存到日志文件
		logging.StreamHandler()  # 输出到控制台
	]
)


class LoggerWriter:
	def __init__(self, level):
		self.level = level

	def write(self, message):
		if message.strip():  # 忽略空消息
			self.level(message)

	def flush(self):  # 兼容性定义
		pass


sys.stdout = LoggerWriter(logging.info)  # 重定向 stdout 到 logging.info
sys.stderr = LoggerWriter(logging.error)  # 重定向 stderr 到 logging.error

# training
DATA_PATH = "D:\\资料\\授课硕\\MATH7224 信息论\\小组作业\\数据集\\mit-bih-arrhythmia-database-1.0.0\\"
FILE_NAME = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111",
             "112", "113", "114", "115", "116", "117", "118", "119","121", "122", "123",
             "124", "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
             "213", "214", "217", "217", "219", "220", "221", "222", "223", "228", "230",
             "231", "232", "233", "234"]
SAMPLE_SIZE = 650000


def get_normal_beat(samples, symbols, start, end):
	indices = np.where(symbols != 'N')[0]
	if indices.size == 0:
		return [[start, end]]
	else:
		ranges = []
		start_pos = start
		for end_pos in indices:
			ranges.append([start_pos, samples[int(end_pos)]])
			if int(end_pos) + 1 >= len(samples):
				break
			start_pos = samples[int(end_pos) + 1]
		ranges.append([start_pos, end])
		return ranges


min_val = -0.5
max_val = 1.0
num_of_bins = [50, 100, 150, 200, 250]
max_depths = range(5, 26, 5)
window_size = 100

random.shuffle(FILE_NAME)
split_point = int(len(FILE_NAME) * 0.3)
train_set = FILE_NAME[:split_point]  # 训练集 (30%)
print("train set:", train_set)

time_record = []
for max_d in max_depths:
	encoder_ML = PDD(max_d)
	encoder_V5 = PDD(max_d)

	for num in num_of_bins:
		time_record.append(0)

		for name in train_set:
			print(max_d, num, name)

			# 抽样20000次采样
			random_start = random.randint(0, SAMPLE_SIZE - 20000)

			data = DATA_PATH + name
			record = wfdb.rdrecord(data, physical=True, )
			annotations = wfdb.rdann(data, 'atr',
			                         sampfrom=random_start, sampto=random_start + 20000)
			indicis = get_normal_beat(annotations.sample, annotations.symbol, random_start, random_start + 20000)
			print(indicis)

			sig_ML = record.p_signal[:, :1].flatten()
			sig_V5 = record.p_signal[:, 1:].flatten()
			d_sig_ML, _ = uniform_quantization(sig_ML, min_val, max_val, num)
			d_sig_V5, _ = uniform_quantization(sig_V5, min_val, max_val, num)

			normal_ML, normal_V5 = [], []
			for index in indicis:
				if index[1] - index[0] > window_size:
					for i in range(index[0], index[1] - window_size):
						normal_ML.append(d_sig_ML[i:i + window_size])
						normal_V5.append(d_sig_V5[i:i + window_size])
				else:
					normal_ML.append(d_sig_V5[index[0]:index[1]])
					normal_V5.append(d_sig_V5[index[0]:index[1]])

			print("start training")
			start_time = time.time()
			for data in normal_ML:
				encoder_ML.train(data)
			for data in normal_V5:
				encoder_V5.train(data)
			end_time = time.time()
			elapsed_time = end_time - start_time
			time_record[-1] += elapsed_time

		print("start building scheme")
		start_time = time.time()
		encoder_ML.build_code_scheme()
		encoder_V5.build_code_scheme()
		end_time = time.time()
		elapsed_time = end_time - start_time
		time_record[-1] += elapsed_time
		print(f"代码运行时间: {time_record[-1]:.3f} 秒")

		encoder_ML.save(f"./model/ML_depth_{max_d}_bins_{num}.json")
		encoder_V5.save(f"./model/V5_depth_{max_d}_bins_{num}.json")
