import wfdb
import matplotlib.pyplot as plt
import numpy as np
import json
from src.pdd_parser import *
from src.utils import *
from src.lz78_parser import *

DATA_PATH = "D:\\资料\\授课硕\\MATH7224 信息论\\小组作业\\数据集\\mit-bih-arrhythmia-database-1.0.0\\"
FILE_NAME = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111",
             "112", "113", "114", "115", "116", "117", "118", "119", "121", "122", "123",
             "124", "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
             "213", "214", "217", "217", "219", "220", "221", "222", "223", "228", "230",
             "231", "232", "233", "234"]
max_D = 5
number_of_bins = 100

MODEL_PATH = "E:\\Jupyter Notebook\\model\\PDA\\"
MODEL_NAME_ML = f"ML_depth_{max_D}_bins_{number_of_bins}.json"
MODEL_NAME_V5 = f"V5_depth_{max_D}_bins_{number_of_bins}.json"

min_val = -0.5
max_val = 1.0
window_size = 100


def extract_N_beats(window_size=50):
	name = '100'
	record = wfdb.rdrecord(DATA_PATH + name)
	annotation = wfdb.rdann(DATA_PATH + name, 'atr', sampto=5000)

	raw_signal_0 = record.p_signal[:, 0]  # 获得第一个导联的信号
	signal_0, _ = uniform_quantization(raw_signal_0,
	                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)
	raw_signal_1 = record.p_signal[:, 1]
	signal_1, _ = uniform_quantization(raw_signal_1,
	                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)
	N_beat_segments_0 = []
	N_beat_segments_1 = []
	for i, (sample, label) in enumerate(zip(annotation.sample, annotation.symbol)):
		if label == 'N':  # 找到N标注的心拍
			start_idx = sample - window_size  # 前500个采样点
			end_idx = sample + window_size  # 后500个采样点

			# 检查是否超出边界
			if start_idx < 0 or end_idx > len(signal_0):
				continue  # 忽略不满足条件的心拍

			# 保存满足条件的片段和标注
			N_beat_segments_0.append(signal_0[start_idx:end_idx])
			N_beat_segments_1.append(signal_1[start_idx:end_idx])

	return N_beat_segments_0, N_beat_segments_1


def extract_non_N_beats(window_size=50):
	"""
	提取所有非N标注心拍及其前后500次采样点。
	如果前后采样点不足，则忽略该心拍。

	Parameters:
	- window_size: 前后采样点的数量（50）

	Returns:
	- non_N_beat_segments: 一个列表，每个元素是包含前后500个采样点的片段
	- non_N_annotations: 与片段对应的标注类型
	"""

	non_N_beat_segments_0, non_N_beat_segments_1 = [], []
	non_N_annotations = []
	for name in FILE_NAME:
		# 读取信号和标注
		record = wfdb.rdrecord(DATA_PATH + name)
		annotation = wfdb.rdann(DATA_PATH + name, 'atr')

		raw_signal_0 = record.p_signal[:, 0]  # 获得第一个导联的信号
		signal_0, _ = uniform_quantization(raw_signal_0,
		                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)
		raw_signal_1 = record.p_signal[:, 1]
		signal_1, _ = uniform_quantization(raw_signal_1,
		                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)

		# 遍历所有标注
		for i, (sample, label) in enumerate(zip(annotation.sample, annotation.symbol)):
			if label != 'N':  # 找到非N标注的心拍
				start_idx = sample - window_size  # 前500个采样点
				end_idx = sample + window_size  # 后500个采样点

				# 检查是否超出边界
				if start_idx < 0 or end_idx > len(signal_0):
					continue  # 忽略不满足条件的心拍

				# 保存满足条件的片段和标注
				non_N_beat_segments_0.append(signal_0[start_idx:end_idx])
				non_N_beat_segments_1.append(signal_1[start_idx:end_idx])
				non_N_annotations.append(label)

	return non_N_beat_segments_0, non_N_beat_segments_1, non_N_annotations


# 使用示例
if __name__ == "__main__":
	# LZ78 pretrain
	print("lz78 pretraining...")
	name = '100'
	record = wfdb.rdrecord(DATA_PATH + name)
	annotation = wfdb.rdann(DATA_PATH + name, 'atr')

	raw_signal_0 = record.p_signal[:, 0]  # 获得第一个导联的信号
	signal_0, _ = uniform_quantization(raw_signal_0,
	                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)
	raw_signal_1 = record.p_signal[:, 1]
	signal_1, _ = uniform_quantization(raw_signal_1,
	                                   min_value=min_val, max_value=max_val, num_bins=number_of_bins)

	lz78_encoder_ML = LZ78()
	lz78_encoder_ML.train(signal_0)
	lz78_encoder_V5 = LZ78()
	lz78_encoder_V5.train(signal_1)

	print("data preparing...")
	with open("data.json", "r") as file:
		data_dict = json.load(file)
		N_segments_0 = data_dict['N_segments_0']
		N_segments_1 = data_dict['N_segments_1']
		non_N_segments_0 = data_dict['non_N_segments_0']
		non_N_segments_1 = data_dict['non_N_segments_1']
		non_N_labels = data_dict['non_N_labels']

	non_N_0 = dict()
	non_N_1 = dict()
	for label in non_N_labels:
		if label not in non_N_0:
			non_N_0[label] = []
			non_N_1[label] = []

	# N_segments_0, N_segments_1 = extract_N_beats()
	# non_N_segments_0, non_N_segments_1, non_N_labels = extract_non_N_beats()
	# data_dict = {
	# 	"N_segments_0": N_segments_0,
	# 	"N_segments_1": N_segments_1,
	# 	"non_N_segments_0": non_N_segments_0,
	# 	"non_N_segments_1": non_N_segments_1,
	# 	"non_N_labels": non_N_labels
	# }
	# with open("data.json", "w") as file:
	# 	json.dump(data_dict, file)

	print("loading...")
	encoder_ML = PDD(max_D)
	encoder_ML.load(MODEL_PATH + MODEL_NAME_ML)
	encoder_V5 = PDD(max_D)
	encoder_V5.load(MODEL_PATH + MODEL_NAME_V5)

	print("calculate non N average score...")
	for segment_0, segment_1, label in zip(non_N_segments_0, non_N_segments_1, non_N_labels):
		c_subseq_pd = encoder_ML.parse(segment_0)
		c_subseq_lz78 = lz78_encoder_ML.encode(segment_0)
		score_0 = encoder_ML.get_total_bits(c_subseq_pd) - lz78_encoder_ML.get_total_bits(c_subseq_lz78)
		non_N_0[label].append(score_0)

		c_subseq_pd = encoder_ML.parse(segment_1)
		c_subseq_lz78 = lz78_encoder_ML.encode(segment_1)
		score_1 = encoder_ML.get_total_bits(c_subseq_pd) - lz78_encoder_ML.get_total_bits(c_subseq_lz78)
		non_N_1[label].append(score_1)

	print(non_N_0)
	non_N_0_means = {key: sum(values) / len(values) for key, values in non_N_0.items()}
	non_N_1_means = {key: sum(values) / len(values) for key, values in non_N_1.items()}

	print("calculate N average score...")
	normal_num = len(N_segments_0)
	normal_score_0 = []
	normal_score_1 = []

	for segment_0, segment_1 in zip(N_segments_0, N_segments_1):
		c_subseq_pd = encoder_ML.parse(segment_0)
		c_subseq_lz78 = lz78_encoder_ML.encode(segment_0)
		score_0 = encoder_ML.get_total_bits(c_subseq_pd) - lz78_encoder_ML.get_total_bits(c_subseq_lz78)
		normal_score_0.append(score_0)

		c_subseq_pd = encoder_ML.parse(segment_1)
		c_subseq_lz78 = lz78_encoder_ML.encode(segment_1)
		score_1 = encoder_ML.get_total_bits(c_subseq_pd) - lz78_encoder_ML.get_total_bits(c_subseq_lz78)
		normal_score_1.append(score_1)

	n_score_0 = sum(normal_score_0) / normal_num
	n_score_1 = sum(normal_score_0) / normal_num

	print(n_score_0, n_score_1)

	non_N_0_means['N'] = n_score_0
	non_N_1_means['N'] = n_score_1

	print("non_N_0 均值:", non_N_0_means)
	print("non_N_1 均值:", non_N_1_means)

	# 提取 keys 和 values
	keys = list(non_N_0_means.keys())  # 横坐标的 key
	values_0 = list(non_N_0_means.values())  # non_N_0 的均值
	values_1 = list(non_N_1_means.values())  # non_N_1 的均值

	x = range(len(keys))  # 横坐标位置
	bar_width = 0.4  # 柱的宽度
	plt.bar(x, values_0, bar_width, label='non_N_0_means', color='blue')  # non_N_0 的柱状图
	plt.bar([i + bar_width for i in x], values_1, bar_width, label='non_N_1_means', color='orange')  # non_N_1 的柱状图
	plt.xlabel('Keys')  # 横坐标标签
	plt.ylabel('Mean Values')  # 纵坐标标签
	plt.title('Mean Values for Each Key')  # 图标题
	plt.xticks([i + bar_width / 2 for i in x], keys)  # 将 x 轴刻度设置为 keys 对应的位置
	plt.legend()
	plt.show()
