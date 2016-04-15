import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dowjones_file_path = 'C:/Users/dano/Desktop/MachineLearningAlgorithms/Datasets/Dow Jones Index/dow_jones_index.csv'

data = pd.read_csv(dowjones_file_path, parse_dates=[2])

def process_dollar_to_float(column):
	new_array = np.array([])
	for value in column:
		if value == 'NaN':
			converted_val = 0
			new_array = np.append(new_array, converted_val)
		if value[0] == '$':
			converted_val = float(value.replace('$',''))
			new_array = np.append(new_array, converted_val)
		else:
			try:
				converted_val = float(value)
				new_array = np.append(new_array, converted_val)
			except:
				print "error"
	return new_array

def process_string_to_float(column):
	new_array = np.array([])
	for value in column:
		converted_val = float(value)
		new_array = np.append(new_array, converted_val)
	return new_array

#TODO: more efficiency with this
opened = process_dollar_to_float(data['open'])
highest = process_dollar_to_float(data['high'])
lowest = process_dollar_to_float(data['low'])
closed = process_dollar_to_float(data['close'])
nxt_close = process_dollar_to_float(data['next_weeks_close'])
nxt_open = process_dollar_to_float(data['next_weeks_open'])

volume = process_string_to_float(data['volume'])
return_nxt_div = process_string_to_float(data['percent_return_next_dividend'])
days_to_div = process_string_to_float(data['days_to_next_dividend'])
perc_change_prc = process_string_to_float(data['percent_change_price'])
perc_change_vol = process_string_to_float(data['percent_change_volume_over_last_wk'])
prev_weeks_vol = process_string_to_float(data['previous_weeks_volume'])
perc_change_nxtwk = process_string_to_float(data['percent_change_next_weeks_price'])

quarters = data['quarter']

feature_array = [opened, highest, lowest, closed, volume, perc_change_prc, \
	nxt_open, nxt_close, perc_change_nxtwk, days_to_div]

clstr = KMeans(n_clusters=2, init='k-means++')
clstr.fit(feature_array)

centers = clstr.cluster_centers_

plot = plt.scatter(feature_array[4], feature_array[9], \
	c=return_nxt_div, alpha=0.8)

plt.xlabel('Volume Sold')
plt.ylabel('Lowest Value')
plt.show()




