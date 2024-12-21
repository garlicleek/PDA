from utils import *
import math


class LZ78:
	def __init__(self):
		self.dictionary = {(): 0}
		self.current_dict_size = 1

	def train(self, data):
		w = []
		for c in data:
			w_c = w + [c]
			w_c_tuple = tuple(w_c)

			if w_c_tuple in self.dictionary:
				w = w_c
			else:
				self.dictionary[w_c_tuple] = self.current_dict_size
				self.current_dict_size += 1
				w = []

	def encode(self, data):
		encoded_result = []
		w = []

		temp_dict = self.dictionary.copy()
		temp_dict_size = self.current_dict_size

		dict_size_record = []  # 记录编码过程中字典大小的变化
		current_size = temp_dict_size

		for c in data:
			w_c = w + [c]
			w_c_tuple = tuple(w_c)

			if w_c_tuple in temp_dict:
				w = w_c
			else:
				prefix_index = temp_dict.get(tuple(w), 0)
				encoded_result.append((prefix_index, c))
				dict_size_record.append(current_size)  # 记录当前编码时的字典大小
				temp_dict[w_c_tuple] = current_size
				current_size += 1
				w = []

		if w:
			prefix_index = temp_dict.get(tuple(w), 0)
			encoded_result.append((prefix_index, None))
			dict_size_record.append(current_size)

		return encoded_result, dict_size_record

	def get_total_bits(self, encoded_result):
		"""Calculate total bits using progressive dictionary size"""

		if not isinstance(encoded_result, tuple):
			return 0

		encoded_phrases, _ = encoded_result
		if not encoded_phrases:
			return 0

		c = len(encoded_phrases)  # 短语数量
		total_bits = c * (math.log2(c) + 1)
		return total_bits


if __name__ == "__main__":
	# Preprocessing
	case = TEST_CASES['loop']
	N = len(case)
	min_val = min(case)
	max_val = max(case)
	number_of_bins = 100
	print(f"preprocess: min={min_val}, max={max_val}, #bins={number_of_bins}")

	# Discretize the samples
	d_x, _ = uniform_quantization(case, min_val, max_val, number_of_bins)

	# Calculate bin size
	bin_size = (max_val - min_val) / number_of_bins

	# Use LZ78 for encoding
	encoder = LZ78()
	encoder.train(d_x)
	c_x1 = encoder.encode(d_x[:10])
	c_x2 = encoder.encode(d_x[10:])
	print(encoder.get_total_bits(c_x1), encoder.get_total_bits(c_x2))

	encoder.__init__()
	c_x1 = encoder.encode(d_x[:10])
	c_x2 = encoder.encode(d_x[10:])
	print(encoder.get_total_bits(c_x1), encoder.get_total_bits(c_x2))
