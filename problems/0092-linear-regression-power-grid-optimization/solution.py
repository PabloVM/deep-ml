import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	y = np.array(consumption_data)
	X =  np.arange(1, len(y) + 1)
	y = y - 10 * np.sin(2 * PI * X /10)
	X = np.column_stack((X,np.ones(len(X))))
	XT = X.transpose()
	w = np.linalg.inv(XT @ X) @ XT @ y
	m, b = w[0] , w[1]

	pred = (m * 15 + b) + 10 * np.sin(2 * PI * 15 /10)
	return int(1.05 * np.ceil(pred))
	