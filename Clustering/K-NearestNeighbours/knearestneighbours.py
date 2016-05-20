import numpy as np
import theano.tensor as T
import theano

rand_gen = np.random

def calc_distance(point1, point2):
	if point1.shape != point2.shape:
		print ("Shapes not equal")
	print(point1, point2)
	final_dist = 0

	for feat1, feat2 in zip(point1, point2):
		op = (feat1 - feat2) ** 2
		final_dist += op

	return np.sqrt(final_dist)


input_var = T.vector('input')
nb = T.matrix('neighbours')

k = T.iscalar('k_value')
seq_count = T.iscalar('sequnces_range')

seq = T.arange(seq_count)

output = T.as_tensor_variable(np.asarray(0, seq.dtype))

scan_result, scan_updates = theano.scan(fn=calc_distance, outputs_info=output, sequences=seq)

knn = theano.function(inputs=[input_var, nb, k], outputs_info=scan_result)






