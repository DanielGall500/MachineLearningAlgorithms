from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

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

print frequency_table

likelihood_table = np.array([])

targs = iris.target

for d in frequency_table:
	idx = 0 #find better iteration method
	likelihood_storage = {}
	for key, value in d.iteritems():
		prob = float(value) / float(total_freq[0])
		likelihood_storage[key] = [round(prob, 2), targs[idx]]
		idx += 1
		print targs[idx], idx
	likelihood_table = np.append(likelihood_table, likelihood_storage)

from collections import OrderedDict

likelihood_table = [OrderedDict(sorted(d.items())) for d in likelihood_table]

likelihood_dataframe = pd.DataFrame()
















