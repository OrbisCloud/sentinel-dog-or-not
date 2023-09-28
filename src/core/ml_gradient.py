from typing import List
from dto.ml_matrix_base import Matrix, transpose_
from dto.ml_hypothesis import Hypothesis
from dto.ml_theta import Theta, create_empty_theta
from dto.ml_predictions import Predictions
from dto.ml_network import Network
from core.ml_sigmoid import calculate_sigmoids, calculate_sigmoid


def calculate_predictions(theta: Matrix, hypothesis: Matrix, metrics=False) -> Predictions:
	# This function performs the following calculation:

	# Let Theta =   [t1]
	#               [t2]
	#               [t3]

	# Let X =       [x1]
	#               [x2]
	#               [x3]

	# Let Theta(transpose) =    [t1, t2, t3]

	# Theta(transpose) * X = 
	#
	# [t1, t2, t3] *    [x1] = [t1x1 + t2x2 + t3x3]
	#                   [x2]
	#                   [x3]
	
	# z stands for theta_transpose_x
	# a stands for the new layer predictions
	
	theta_transpose_x: Matrix = hypothesis * theta
	predictions: Matrix = theta_transpose_x.apply(calculate_sigmoid)
	predictions.name = f"predictions, ({predictions.name})"
			
	# only to be used for diagnostics
	if metrics:
		print(predictions)
		
	return predictions

		
def calculate_gradient_descent(hypothesis: Hypothesis, theta: Matrix, predictions: Matrix, metrics=False, m_cost=None) -> Theta:

	# This function calculates a gradient.
	# A gradient consists of slopes.
	# The slopes are retrieved from a derivative of a squared cost function.
	# cost(x) = (1/2n) * sum => (theta_transpose_x - actual_y)^2
	# d/dTheta = (1/n) * sum => (theta_transpose_x - actual_y) * x(i)(j)
	
	# vectorized to the extent that it can apply all theta values at the same time (e.g. for multiple classifiers)	
	#for i in range(0, theta.width):
	
	if not m_cost:
		m_cost: Matrix = predictions - hypothesis.mat_y()	
	e_m_gradients: Matrix = transpose_(m_cost) * transpose_(hypothesis)
	e_min_js: Matrix = (-theta.alpha / hypothesis.width) * e_m_gradients	
	new_theta: Matrix = theta + e_min_js
		
	# only to be used for diagnostics
	if metrics:
		print(predictions)
		print(hypothesis.mat_y())
		print(m_cost)
		
		print(transpose_(m_cost))
		print(transpose_(hypothesis))
		
		print(e_m_gradients)
		print(e_min_js)
		print(new_theta)
		
	theta.points = new_theta.points
	return theta

