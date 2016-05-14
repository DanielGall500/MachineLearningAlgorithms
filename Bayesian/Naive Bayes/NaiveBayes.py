from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

features = iris.data
target = iris.target

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

import theano
import theano.tensor as T

frequency_table = np.array([])
total_freq = []

for col in features.T:
	frequencies = {}
	for val in col:
		val = round(val, 3)
		if val in frequencies:
			frequencies[val] = frequencies[val] + 1
		else:
			frequencies[val] = 1
	frequency_table = np.append(frequency_table, frequencies)
	total_freq.append(sum(frequencies.itervalues()))

likelihood_table = np.array([])

for d in frequency_table:
	likelihood_storage = {}
	for key, value in d.iteritems():
		prob = float(value) / float(total_freq[0])
		likelihood_storage[key] = round(prob, 2)
	likelihood_table = np.append(likelihood_table, likelihood_storage)

print likelihood_table












