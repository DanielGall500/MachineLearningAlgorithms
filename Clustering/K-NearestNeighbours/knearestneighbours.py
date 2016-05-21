import numpy as np
import theano.tensor as T
import theano

rand_gen = np.random

def calc_distances(step, inputs, neighbours, k):

	nb_feat = neighbours[step] #everything but last feature
	target = neighbours[-1] #last feature

	print 'Input features: {}'.format(inputs)
	print 'Neighbour features: {}'.format(nb_feat)

	if inputs.shape != nb_feat.shape:
		print("Shapes Not Equal: Inputs - {}, Neighbours - {}".format(inputs.shape, nb_feat.shape))

	distances = T.sqrt((inputs - nb_feat) ** 2)

	return step #distances #ERROR IS UP HERE

input_var = T.matrix('input')
nb = T.matrix('neighbours')

k = T.iscalar('k_value')

step_count = T.iscalar('steps')

seq = T.arange(step_count)

output_model = T.as_tensor_variable(np.asarray(0, step_count.dtype))

scan_result, scan_updates = theano.scan(fn=calc_distances, outputs_info=None, sequences=seq, non_sequences=[input_var, nb, k])

k_nearest_neighbours = theano.function(inputs=[input_var, nb, k, step_count], outputs=scan_result)


#Create samples for testing

input_features = np.array([[5.1, 3.8, 1.5, 0.3]])

k = 5

neighbours = np.array([ # 0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica
[5.1,3.5,1.4,0.3,0],[5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],
[5.4,3.4,1.7,0.2,0],[5.1,3.7,1.5,0.4,0],[4.6,3.6,1.0,0.2,0],
[5.1,3.3,1.7,0.5,0],[4.8,3.4,1.9,0.2,0],[5.0,3.0,1.6,0.2,0],
[5.0,3.4,1.6,0.4,0],[6.0,3.4,4.5,1.6,1],[6.7,3.1,4.7,1.5,1],
[6.3,2.3,4.4,1.3,1],[5.6,3.0,4.1,1.3,1],[5.5,2.5,4.0,1.3,1],
[5.5,2.6,4.4,1.2,1],[6.1,3.0,4.6,1.4,1],[5.8,2.6,4.0,1.2,1],
[5.0,2.3,3.3,1.0,1],[5.6,2.7,4.2,1.3,1],[6.9,3.2,5.7,2.3,2],
[5.6,2.8,4.9,2.0,2],[7.7,2.8,6.7,2.0,2],[6.3,2.7,4.9,1.8,2],
[6.7,3.3,5.7,2.1,2],[7.2,3.2,6.0,1.8,2],[6.2,2.8,4.8,1.8,2],
[6.1,3.0,4.9,1.8,2],[6.4,2.8,5.6,2.1,2],[7.2,3.0,5.8,1.6,2]])

steps = len(neighbours)

knn = k_nearest_neighbours(input=input_features, neighbours=neighbours, k_value=k, steps=steps)
print knn






