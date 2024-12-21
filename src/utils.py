

def uniform_quantization(samples, min_value, max_value, num_bins):
	"""
    Truncate sample data to specified interval, then discretize it.
	Args:
		samples: sample data
		min_value: minimum of target interval
		max_value: maximum of target interval
		num_bins: Degree of dispersion

	Returns:
		discretized_samples: discretization sequence
	"""

	bin_size = (max_value - min_value) / num_bins
	bin_centers = [min_value + bin_size * (i + 0.5) for i in range(num_bins)]

	discretized_samples = []
	for sample in samples:
		sample = max(min(sample, max_value), min_value)
		bin_index = min(int((sample - min_value) / bin_size), num_bins - 1)
		discretized_samples.append(round(bin_centers[bin_index], 6))

	return discretized_samples, bin_centers


TEST_CASES = {
	'basic': [1.2, 2.5, 3.7, 4.1, 5.6, 6.8, 7.9, 8.0, 9.1, 10.2],
	'loop': [1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5,
	         1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5,
	         3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7,
	         1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5,
	         1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5, 1.2, 2.5, 3.7, 2.5,
	         ]
}

if __name__ == '__main__':
	# Discretize the samples
	case = TEST_CASES['basic']
	discrete_series, _ = uniform_quantization(case, min(case), max(case), 10)
	print(discrete_series)
