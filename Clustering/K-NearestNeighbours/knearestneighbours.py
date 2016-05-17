import numpy as np
import theano.tensor as T
import theano

rand_gen = np.random

def calc_distance(point1, point2):
	if len(point1) != len(point2):
		print ("Shapes not equal")

	final_dist = 0

	for feat1, feat2 in zip(point1, point2):
		op = (feat1 - feat2) ** 2
		final_dist += op

	return np.sqrt(final_dist)


X = T.matrix('x')
Y = T.vector('y')

k = 3

seq = theano.tensor.arange(X)
output = T.as_tensor_variable(np.asarray(0, seq.dtype))







