import numpy as np
import json

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from huffman import *
from utils import *
import math


class PDD:
	def __init__(self, max_D, dataFlow=False):
		self.max_D = max_D  # maximum code depths
		self.pattern_dict = dict()  # {code_depth: patterns}
		self.code_scheme = dict()  # {code_depth: code schemes}

	def save(self, path):
		pattern_dict_serializable = {
			code_depth: {str(k): v for k, v in patterns.items()}
			for code_depth, patterns in self.pattern_dict.items()
		}

		code_scheme_serializable = {
			code_depth: {
				'codes': {str(k): v for k, v in scheme['codes'].items()},
				'lengths': {str(k): v for k, v in scheme['lengths'].items()}
			}
			for code_depth, scheme in self.code_scheme.items()
		}

		data = {
			'max_D': self.max_D,
			'pattern_dict': pattern_dict_serializable,
			'code_scheme': code_scheme_serializable
		}
		# 将数据写入 JSON 文件
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(data, f, ensure_ascii=False, indent=4)  # 格式化输出
		print(f"PDD 对象已成功保存到 {path}")

	def load(self, path):
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
			self.max_D = data['max_D']

			self.pattern_dict = {
				int(code_depth): {tuple(eval(k)): v for k, v in patterns.items()}
				for code_depth, patterns in data['pattern_dict'].items()
			}

			self.code_scheme = {
				int(code_depth): {
					'codes': {tuple(eval(k)): v for k, v in scheme['codes'].items()},
					'lengths': {tuple(eval(k)): v for k, v in scheme['lengths'].items()}
				}
				for code_depth, scheme in data['code_scheme'].items()
			}

		print(f"PDD 对象已成功从 {path} 加载")

	def train(self, source):
		"""build pattern dictionary"""
		code_depth = 1
		while code_depth <= self.max_D:
			self._train_depth(source, code_depth)
			code_depth += 1

	def init_alphabet(self, alphabet):
		"""initialize full alphabet"""
		code_depth = 1
		if code_depth not in self.pattern_dict:
			self.pattern_dict[code_depth] = {}
		# 对于code_depth=1时，需要ALPHABET完全补足
		self.pattern_dict[code_depth] = {tuple([key]): 0 for key in alphabet}

	def build_code_scheme(self):
		"""build huffman coding tree"""
		code_depth = 1
		while code_depth <= self.max_D:
			pattern_tb = self.pattern_dict[code_depth]
			codes = [code for code in pattern_tb.keys()]
			freqs = np.array([freq for freq in pattern_tb.values()])
			code_dict = build_huffman_tree(codes, freqs)

			# 记录每个短语的码字长度
			code_lengths = {phrase: len(code) for phrase, code in code_dict.items()}

			# 存储编码方案和码字长度
			self.code_scheme[code_depth] = {'codes': code_dict, 'lengths': code_lengths}
			code_depth += 1

	def _train_depth(self, source, code_depth):
		"""get pattern dict with given code_depth"""
		if code_depth not in self.pattern_dict:
			self.pattern_dict[code_depth] = {}

		pattern_dict = self.pattern_dict[code_depth]
		for start in range(len(source) - code_depth + 1):
			substr = tuple(source[start:start + code_depth])  # use tuple as key
			if substr not in pattern_dict:
				pattern_dict[substr] = 1
			else:
				pattern_dict[substr] += 1
		self.pattern_dict[code_depth] = pattern_dict

	def parse(self, data):
		encoded = []

		v_c = 0
		depth = 1

		while v_c + depth - 1 < len(data):
			phrase = tuple(data[v_c:v_c + depth])
			if phrase in self.pattern_dict[depth]:
				if depth < self.max_D:
					depth += 1
				else:
					code_info = self.code_scheme[depth]
					code_length = code_info['lengths'][phrase]
					encoded.append((phrase, code_length))

					v_c += depth
					depth = 1
			else:
				phrase = tuple(data[v_c:v_c + depth - 1])
				code_info = self.code_scheme[depth]
				code_length = code_info['lengths'][phrase]
				encoded.append((phrase, code_length))

				v_c += depth - 1
				depth = 1

		# 处理最后的短语
		if v_c < len(data):
			phrase = tuple(data[v_c:])
			code_info = self.code_scheme[len(phrase)]
			code_length = code_info['lengths'][phrase]
			encoded.append((phrase, code_length))

		return encoded  # 返回 (phrase, code_length) 的列表

	def get_total_bits(self, encoded_result):
		total_bits = 0
		c = len(encoded_result)
		for _, code_length in encoded_result:
			total_bits += code_length  # 累加每个短语的编码长度
		total_bits += c * math.log2(self.max_D)  # 加上每个短语指示深度的位数
		return total_bits


if __name__ == "__main__":
	encoder = PDD(max_D=3)
	# 首先兜底完整的ALPHABET
	case = TEST_CASES['basic']
	min_val = min(case)
	max_val = max(case)
	number_of_bins = 10
	bin_size = (max_val - min_val) / number_of_bins
	alphabet = [round(min_val + bin_size * (i + 0.5), 4) for i in range(number_of_bins)]
	encoder.init_alphabet(alphabet)
	# 对于每个序列独立调用train，生成或更新PDD，记录当前频率
	encoder.train([1.65, 2.55, 3.45, 4.35, 5.25, 7.05, 7.95, 7.95, 8.85, 10.65])
	encoder.train([1.65, 2.55, 3.45, 4.35, 5.25, 7.05, 7.95, 7.95, 8.85, 10.65])
	encoder.train([1.65, 2.55, 3.45, 4.35, 5.25, 7.05, 7.95, 7.95, 8.85, 10.65])
	# 遍历完整train data之后，基于当前频率构建code tree
	encoder.build_code_scheme()
	# 使用code tree完成编码，完成测试
	res = encoder.parse([4.35, 5.25, 7.95, 7.05])
	print(encoder.pattern_dict)
	encoder.save(f"./encoder.json")
	encoder.load(f"./encoder.json")
	print(encoder.pattern_dict)