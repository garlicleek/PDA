import matplotlib.pyplot as plt
import numpy as np

# 数据
# Precision & Recall about num_of_bins when max_d = 10
max_depths = range(5, 26, 5)

# Data for precision & recall about max_d when num_of_bins = 100
num_of_bins = [50, 100, 150, 200, 250]


# 计算 F1-score
def calculate_f1(precision, recall):
	return [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]


def plt_F1():
	precision_depth = [0.662, 0.642, 0.630, 0.628, 0.625]
	recall_depth = [0.835, 0.840, 0.850, 0.848, 0.850]
	precision_bins = [0.692, 0.642, 0.630, 0.562, 0.558]
	recall_bins = [0.732, 0.840, 0.860, 0.882, 0.877]
	# F1-score about num_of_bins
	f1_bins = calculate_f1(precision_bins, recall_bins)

	# F1-score about max_d
	f1_depth = calculate_f1(precision_depth, recall_depth)

	# 创建 2 个子图
	plt.figure(figsize=(14, 6))

	# 子图 1: Precision, Recall & F1 vs num_of_bins
	plt.subplot(1, 2, 1)
	plt.plot(num_of_bins, precision_bins, marker='o', label='Precision', color='blue')
	plt.plot(num_of_bins, recall_bins, marker='s', label='Recall', color='green')
	plt.plot(num_of_bins, f1_bins, marker='^', label='F1-score', color='red')
	plt.title("Precision, Recall & F1 vs num_of_bins (max_d=10)", fontsize=14)
	plt.xlabel("num_of_bins", fontsize=12)
	plt.ylabel("Score of V5 Channel", fontsize=12)
	plt.ylim(0.5, 0.9)  # 设置 Y 轴范围
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend(fontsize=10)

	# 子图 2: Precision, Recall & F1 vs max_d
	plt.subplot(1, 2, 2)
	plt.plot(max_depths, precision_depth, marker='o', label='Precision', color='blue')
	plt.plot(max_depths, recall_depth, marker='s', label='Recall', color='green')
	plt.plot(max_depths, f1_depth, marker='^', label='F1-score', color='red')
	plt.title("Precision, Recall & F1 vs max_d (num_of_bins=100)", fontsize=14)
	plt.xlabel("max_d", fontsize=12)
	plt.ylabel("Score of V5 Channel", fontsize=12)
	plt.ylim(0.5, 0.9)  # 设置 Y 轴范围
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend(fontsize=10)

	# 调整布局并显示
	plt.tight_layout()
	plt.show()


def plt_cost():
	# Precision & Recall about num_of_bins when max_d = 10
	max_depths = range(5, 26, 5)

	# Data for precision & recall about max_d when num_of_bins = 100
	num_of_bins = [50, 100, 150, 200, 250]
	# 数据
	time_dep = [169.319, 618.505, 823.429, 1231.328, 1649.697]  # 时间 (秒)
	time_bin = [501.290, 544.021, 618.505, 749.695, 815.224]  # 时间 (秒)

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(max_depths, time_dep, marker='o', color='blue')
	plt.title('Training Time vs max_depths (Bins = 100)', fontsize=14)
	plt.xlabel('max_depths', fontsize=12)
	plt.ylabel('Training Time (seconds)', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.subplot(1, 2, 2)
	plt.plot(num_of_bins, time_bin, marker='s', color='orange')
	plt.title('Training Time vs Num_of_bins (Max_d = 10)', fontsize=14)
	plt.xlabel('Number of Bins', fontsize=12)
	plt.ylabel('Training Time (seconds)', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.tight_layout()
	plt.show()

	size_dep = [76044, 3567059, 8625081, 37875243, 103764128]  # 数据大小 (KB)
	size_bin = [999900, 3576059, 7705010, 13015548, 28243739]  # 数据大小 (KB)
	# Convert size from KB to GB
	size_dep_gb = np.array(size_dep) / (1024 * 1024)  # Convert to GB
	size_bin_gb = np.array(size_bin) / (1024 * 1024)  # Convert to GB

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(max_depths, size_dep_gb, marker='o', color='blue')
	plt.title('Training Time vs max_depths (Bins = 100)', fontsize=14)
	plt.xlabel('max_depths', fontsize=12)
	plt.ylabel('Diction Size (GB)', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.subplot(1, 2, 2)
	plt.plot(num_of_bins, size_bin_gb, marker='s', color='orange')
	plt.title('Training Time vs Num_of_bins (Max_d = 10)', fontsize=14)
	plt.xlabel('Number of Bins', fontsize=12)
	plt.ylabel('Diction Size (GB)', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	plt_cost()
