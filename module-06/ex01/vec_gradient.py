import numpy as np
from tools import add_intercept
from prediction import predict_

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.array, without any for loop.
	The three arrays must have compatible shapes.
	Args:
	x: has to be a numpy.array, a matrix of shape m * 1.
	y: has to be a numpy.array, a vector of shape m * 1.
	theta: has to be a numpy.array, a 2 * 1 vector.
	Return:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta is an empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	print(np.average(np.transpose(add_intercept(x))))
	print()
	print((np.array(x * theta) - y))
	exit()
	return np.array([np.average(np.transpose(add_intercept(x)) * (add_intercept(x) * theta - y))])

if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	print(gradient(x, y, theta1))
	# Output:
	# array([[-19.0342574], [-586.66875564]])
	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(gradient(x, y, theta2))
	# Output:
	# array([[-57.86823748], [-2230.12297889]])
