import numpy as np

def add_intercept(x):
	"""Adds a column of 1's to the non-empty numpy.array x.
	Args:
	x: has to be a numpy.array of dimension m * n.
	Returns:
	X, a numpy.array of dimension m * (n + 1).
	None if x is not a numpy.array.
	None if x is an empty numpy.array.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, (np.ndarray, np.generic)):
		return None
	if not x.size:
		return None
	if x.ndim == 1:
		x = x[:,np.newaxis]
	ones = np.ones((len(x), 1))
	return np.hstack((ones, x))

x = np.arange(1,6)
print(add_intercept(x))

y = np.arange(1,10).reshape((3,3))
print(add_intercept(y))
