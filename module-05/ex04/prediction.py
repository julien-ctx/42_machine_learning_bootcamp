import numpy as np
from tools import add_intercept

def predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.array.
	Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
	y_hat as a numpy.array, a vector of dimension m * 1.
	None if x and/or theta are not numpy.array.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not appropriate.
	Raises:
	This function should not raise any Exceptions.
	"""
	if not isinstance(x, (np.ndarray, np.generic)) or not isinstance(theta, (np.ndarray, np.generic)):
		return None
	if not x.size or not theta.size:
		return None
	return np.array(np.sum(add_intercept(x) * theta, axis=1))


if __name__ == "__main__":
	x = np.arange(1,6)

	theta1 = np.array([5, 0])
	print(predict_(x, theta1))

	theta2 = np.array([0, 1])
	print(predict_(x, theta2))

	theta3 = np.array([5, 3])
	print(predict_(x, theta3))

	theta4 = np.array([-3, 1])
	print(predict_(x, theta4))
