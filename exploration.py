import wfdb
import matplotlib.pyplot as plt
import random

from src.pdd_parser import *
from src.utils import *
from src.lz78_parser import *

DATA_PATH: str = "D:\\资料\\授课硕\\MATH7224 信息论\\小组作业\\数据集\\mit-bih-arrhythmia-database-1.0.0\\"
FILE_NAME = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111",
             "112", "113", "114", "115", "116", "117", "118", "119","121", "122", "123",
             "124", "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
             "213", "214", "217", "217", "219", "220", "221", "222", "223", "228", "230",
             "231", "232", "233", "234"]

max_D = 10
number_of_bins = 100

MODEL_PATH = "E:\\Jupyter Notebook\\model\\PDA\\"
MODEL_NAME_ML = f"ML_depth_{max_D}_bins_{number_of_bins}.json"
MODEL_NAME_V5 = f"V5_depth_{max_D}_bins_{number_of_bins}.json"

min_val = -0.5
max_val = 1.0
window_size = 100

random_file = FILE_NAME[random.randint(0, 44)]
random_start = random.randint(0, 650000 - 20000)
data = DATA_PATH + str(random_file)

annotations_sample = wfdb.rdann(data, 'atr',
                                sampfrom=random_start, sampto=random_start + 20000)
non_normal_indices = [[i, sym] for i, sym in zip(annotations_sample.sample, annotations_sample.symbol) if sym != "N"]

record = wfdb.rdrecord(data, physical=True,
                       sampfrom=non_normal_indices[0][0] - 2000, sampto=non_normal_indices[0][0] + 3000)
annotations = wfdb.rdann(data, 'atr', shift_samps=True,
                         sampfrom=non_normal_indices[0][0] - 2000, sampto=non_normal_indices[0][0] + 3000)
indices = [[i, sym] for i, sym in zip(annotations.sample, annotations.symbol)]

sig_ML = record.p_signal[:, :1].flatten()
d_sig_ML, _ = uniform_quantization(sig_ML, min_val, max_val, number_of_bins)
#
print("model loading...")
encoder_ML = PDD(max_D)
encoder_ML.load(MODEL_PATH + MODEL_NAME_ML)

print("lz78 training...")
print(sig_ML, d_sig_ML)
lz78_encoder_ML = LZ78()
lz78_encoder_ML.train(d_sig_ML)

print("scoring...")
score_ML = []
for i_seq in range(0, len(sig_ML) - window_size, 1):
	subseq = d_sig_ML[i_seq:i_seq + window_size]
	c_subseq_pd = encoder_ML.parse(subseq)
	c_subseq_lz78 = lz78_encoder_ML.encode(subseq)
	score = encoder_ML.get_total_bits(c_subseq_pd) - lz78_encoder_ML.get_total_bits(c_subseq_lz78)
	score_ML.append(score)

sig_V5 = record.p_signal[:, 1:].flatten()
d_sig_V5, _ = uniform_quantization(sig_V5, min_val, max_val, number_of_bins)

print("model loading...")
encoder_V5 = PDD(max_D)
encoder_V5.load(MODEL_PATH + MODEL_NAME_V5)

print("lz78 training...")
lz78_encoder_V5 = LZ78()
lz78_encoder_V5.train(d_sig_V5)

print("scoring...")
score_V5 = []
for i_seq in range(0, len(d_sig_V5) - window_size, 1):
	subseq = d_sig_V5[i_seq:i_seq + window_size]
	c_subseq_pd = encoder_V5.parse(subseq)
	c_subseq_lz78 = lz78_encoder_V5.encode(subseq)
	score = encoder_V5.get_total_bits(c_subseq_pd) - lz78_encoder_V5.get_total_bits(c_subseq_lz78)
	score_V5.append(score)

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].plot(score_ML, label='PDA Score')
axs[0].set_xlabel('Sequence Index')
axs[0].set_title('MLII PDA Score')
axs[0].grid(True)
for ano in indices:
	t = ano[0]
	if t < len(score_ML):  # 确保标记点在绘图范围内
		axs[0].plot(t, score_ML[t], 'rx')
		axs[0].text(t, score_ML[t], ano[1],fontsize=15, color='red')
axs[0].legend()

axs[1].plot(score_V5, label='PDA Score')
axs[1].set_xlabel('Sequence Index')
axs[1].set_title('V5 PDA Score')
axs[1].grid(True)
for ano in indices:
	t = ano[0]
	if t < len(score_V5):  # 确保标记点在绘图范围内
		axs[1].plot(t, score_V5[t], 'rx')
		axs[1].text(t, score_V5[t], ano[1],fontsize=15, color='red')
axs[0].legend()

plt.tight_layout()
plt.show()

plt.figure()
wfdb.plot_wfdb(record=record, annotation=annotations)
