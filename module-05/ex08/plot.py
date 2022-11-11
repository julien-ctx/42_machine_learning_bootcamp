import numpy as np
import matplotlib.pyplot as plt

def plot_with_loss(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exception.
	"""
	plt.plot(x, y, 'o')
	plt.plot(x, theta[1] * x + theta[0])
	for item in x:
		plt.plot([x, x], [y, theta[1] * x + theta[0]], 'r:')
	plt.show()

if __name__ == "__main__":
	x = np.arange(1,6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

	theta1= np.array([18, -1])
	plot_with_loss(x, y, theta1)

	theta2 = np.array([14, 0])
	plot_with_loss(x, y, theta2)

	theta3 = np.array([12, 0.8])
	plot_with_loss(x, y, theta3)
