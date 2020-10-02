def series_to_supervised(data, lag_input=1, lag_output=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	
    - data: Sequence of observations as a list or NumPy array.
	- lag_input: Number of lag observations as input (X).
	- lag_output: Number of observations as output (y).
	- dropnan: Boolean whether or not to drop rows with NaN values.
	
    Returns Pandas DataFrame of series framed for supervised learning.
	"""

	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	
	for i in range(lag_input, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	
	for i in range(0, lag_output):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
	data_frame = pd.concat(cols, axis=1)
	data_frame.columns = names
	
	if dropnan:
		data_frame.dropna(inplace=True)
	return data_frame
