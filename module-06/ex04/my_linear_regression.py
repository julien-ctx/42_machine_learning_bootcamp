import numpy as np

class MyLR():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		if len(thetas):
			if not isinstance(thetas[0], list):
				thetas = np.array([[item] for item in thetas])
		self.thetas = thetas

	def add_intercept(self, x):
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

	def predict_(self, x):
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
		if not isinstance(x, (np.ndarray, np.generic)):
			return None
		return np.array(self.add_intercept(x) @ self.thetas)

	@staticmethod
	def loss_elem_(y, y_hat):
		"""
		Description:
		Calculates all the elements (y_pred - y)^2 of the loss function.
		Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
		Returns:
		J_elem: numpy.array, a vector of dimension (number of the training examples,1).
		None if there is a dimension matching problem between X, Y or theta.
		None if any argument is not of the expected type.
		Raises:
		This function should not raise any Exception.
		"""
		return (y_hat - y) ** 2

	@staticmethod
	def loss_(y, y_hat):
		"""
		Description:
		Calculates the value of loss function.
		Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
		Returns:
		J_value : has to be a float.
		None if there is a dimension matching problem between X, Y or theta.
		None if any argument is not of the expected type.
		Raises:
		This function should not raise any Exception.
		"""
		return np.average(MyLR.loss_elem_(y, y_hat)) / 2

	def gradient(self, x, y, theta):
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
		xp = self.add_intercept(x)
		xpt = xp.T
		return np.array((xpt @ (xp @ theta - y)) / y.size)
	
	def fit_(self, x, y):
		"""
		Description:
		Fits the model to the training dataset contained in x and y.
		Args:
		x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
		Returns:
		new_theta: numpy.ndarray, a vector of dimension 2 * 1.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exception.
		"""
		for i in range(self.max_iter):
			self.thetas = self.thetas - self.alpha * self.gradient(x, y, self.thetas)
		return self.thetas

	@staticmethod
	def mse_(y, y_hat):
		"""
		Description:
		Calculate the MSE between the predicted output and the real output.
		Args:
		y: has to be a numpy.array, a vector of dimension m * 1.
		y_hat: has to be a numpy.array, a vector of dimension m * 1.
		Returns:
		mse: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		return np.average((y_hat - y) ** 2)

if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	lr1 = MyLR([2, 0.7])
	# Example 0.0:
	print(lr1.predict_(x))
	# Output:
	# array([[10.74695094],
	# [17.05055804],
	# [24.08691674],
	# [36.24020866],
	# [42.25621131]])
	# Example 0.1:
	print(MyLR.loss_elem_(y, lr1.predict_(x)))
	# Output:
	# array([[710.45867381],
	# [364.68645485],
	# [469.96221651],
	# [108.97553412],
	# [299.37111101]])
	# Example 0.2:
	print(MyLR.loss_(y, lr1.predict_(x)))
	# Output:
	# 195.34539903032385
	# Example 1.0:
	lr2 = MyLR([1, 1], 5e-8, 1500000)
	lr2.fit_(x, y)
	print(lr2.thetas)
	# Output:
	# array([[1.40709365],
	# [1.1150909 ]])
	# Example 1.1:
	print(lr2.predict_(x))
	# Output:
	# array([[15.3408728 ],
	# [25.38243697],
	# [36.59126492],
	# [55.95130097],
	# [65.53471499]])
	# Example 1.2:
	print(MyLR.loss_elem_(y, lr2.predict_(x)))
	# Output:
	# array([[486.66604863],
	# [115.88278416],
	# [ 84.16711596],
	# [ 85.96919719],
	# [ 35.71448348]])
	# Example 1.3:
	print(MyLR.loss_(y, lr2.predict_(x)))
	# Output:
	#80.83996294128525
