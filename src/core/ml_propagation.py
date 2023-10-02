from typing import List
from copy import deepcopy
from dto.ml_matrix_base import Matrix, transpose_
from dto.ml_theta import Theta
from dto.ml_predictions import Predictions
from dto.ml_network import Network
from core.ml_gradient import calculate_predictions
from core.ml_cost import log_cost
from tools.logger import logger


# -~~~~~~~-
# Forward prop
# -~~~~~~~-

# -~~~~~~~-
# This function calculates predictions for every layer
# There are n number of layers, each has w number of weights (theta) to adjust
# For subsequent cost calculations, we only care about the final outcome in layer n
# -~~~~~~~-
def calculate_forward_propagation(network: Network, metrics: bool = False) -> Matrix:
	y_actual: Matrix = network.y_actual
	
	for l in range(0, network.layers-1):
		theta: Theta = network.get_theta(l)
		
		# bias node
		m_bias: List[List[float]] = deepcopy(network.get_layer(l).points)
		[row.append(1) for row in m_bias]
		bias: Matrix = Matrix(m_bias, f"Layer {l}, with bias node added")

		result: Matrix = calculate_predictions(theta, bias)
		network.set_layer(l+1, result)
		
	result: Matrix = network.get_layer(network.layers-1)
	cost: Matrix = result.apply(log_cost, y_actual)
	
	if metrics:
		print(f"---forward prop---")
		print(f"::: y_actual v result")
		print(y_actual)
		print(result)
		
	return network, cost, result
	
	
# -~~~~~~~-
# Back prop
# -~~~~~~~-

# This algorithm calculates the contributory factor of a single weight to the overal cost in / decrease
# It calculates the derivative of the cost function in relation to individual weights.
# -~~~~~~~-

# Delta n-1 -> Theta n-1
# First Delta (for output layer) is calculated differently from every Delta thereafter

# 1 )			Retrieve Data n
# 2 )			Error, for Data n
# 3 )			Derivative of sigmoid function, layer n
# 4 )			Retrieve Data n-1
# 5 )			Delta n-1 -> Multiply Error n with Sigmoid n
# 6 )			Store Delta n-1 in index n-1
# 7 )			Multiply Delta n-1 with Data n-1
# 8 )			CDelta n-1 -> Update Theta layer n-1 with delta layer n-1


# Delta n-i -> Theta n-i where n-i =/ n
# Let Delta n-i = l

# 1	)			Retrieve Data l
# 2	)			Retrieve Delta l+1, Theta l+1
# 3	)			Net l+1 = Delta l+1 with Theta l+1
#	3	)			Calculate Sigmoid l+1
#	4	)			Delta l = Net l+1 with Sigmoid l+1
# 5 )			Store delta l in index l
# 7 )			Multiply Delta l with Data l
# 8 )			CDelta l -> Update Theta l with Delta l (or accumulate)

# -~~~~~~~-
def calculate_back_propagation(network: Network, metrics: bool = False) -> Network:
		
	deltas: List[Matrix] = [0 for l in range(0, network.layers-1)]
	o_layer: Matrix = network.get_layer(network.layers-1)
	i_layer: Matrix = network.get_layer(network.layers-2)
	c_delta: Matrix = network.get_delta(network.layers-2)
	
	d_error: Matrix = -1 * (network.y_actual - o_layer)
	d_sig: Matrix = (o_layer ** (1 - o_layer))
	d_delta: Matrix = d_error ** d_sig
	
	d_delta.name = f"layer {network.layers-2}: Delta"
	deltas[network.layers-2] = d_delta
	
	d_theta: Matrix = transpose_(d_delta) * transpose_(i_layer)
	c_delta = c_delta + d_theta
	network.set_delta(network.layers-2, c_delta)

	for l in reversed(range(0, network.layers-2)):
		i_layer: Matrix = network.get_layer(l)
		o_layer: Matrix = network.get_layer(l+1)
		n_delta: Matrix = deltas[l+1]
		n_theta: Theta = network.get_theta(l+1).range(0, -1)
		c_delta: Matrix = network.get_delta(l)
		
		d_net: Matrix = (n_delta * transpose_(n_theta))
		d_sig: Matrix = (o_layer ** (1 - o_layer))
		d_delta: Matrix = d_net ** d_sig
		
		d_delta.name = f"layer {l}: Delta"
		deltas[l] = d_delta
		
		d_theta: Matrix = transpose_(d_delta) * transpose_(i_layer)
		c_delta = c_delta + d_theta
		network.set_delta(l, c_delta)
			
	if metrics:
		print(f"---back propagation---")
		for l in range(0, network.layers-1):
			d_delta: Matrix = deltas[l]
			c_delta: Matrix = network.get_delta(l)
			
			print(f"::: delta layer {l}")
			print(d_delta)
			print(f"::: cumulative delta layer {l}")
			print(c_delta)
			
	return network
	

# -~~~~~~~-
# Neural descent
# -~~~~~~~-

# -~~~~~~~-
# Gradient descent but for the neural network
# Neural descent is not a thing, I just like to call it that. Sue me
# Actually, don't. Don't sue me.
# -~~~~~~~-
def calculate_neural_descent(network: Network, metrics: bool = False) -> Network:
	
	if metrics:
		print("---calculate neural descent---")
		
	for l in range(0, network.layers-1):
		l_theta: Matrix = network.get_theta(l)
		r_theta: Matrix = l_theta.range(0, -1)
		c_delta: Matrix = network.get_delta(l)
		
		if metrics:
			print(f"::: before theta")
			print(l_theta)
		
		r_theta = r_theta - (network.alpha / len(network.hypotheses)) * c_delta
		a_theta: Matrix = r_theta.add(l_theta.range(l_theta.height-1, l_theta.height))
		network.set_theta(l, a_theta)
		
		if metrics:
			print(f"::: weight adjustment")
			print(c_delta)
			print(f"::: after theta")
			print(a_theta)
				
	return network


# -~~~~~~~-
# Driver for back prop
# -~~~~~~~-

# -~~~~~~~-
# This function executes back prop for every training example.
# It then adjusts weights, and calculates overall outstanding cost.

# TODO
# 1) Gradient checking
# 2) Regularization
# -~~~~~~~-
def propagate(network: Network, metrics: bool = False) -> Network:
	
	network.reset_deltas()
	
	for m in range(0, len(network.hypotheses)):
		network.set_training_example(m)
		network, cost, result = calculate_forward_propagation(network)
		network = calculate_back_propagation(network, metrics=metrics)
	network = calculate_neural_descent(network, metrics=metrics)

	cost: Matrix = Matrix([[0 for i in range(0, network.master_hypothesis.mat_y().height)]], "")
	for m in range(0, len(network.hypotheses)):
	 	network.set_training_example(m)
	 	cost = cost + calculate_forward_propagation(network, metrics=metrics)[1]
	logger.info("::: cost")
	cost = (1 / len(network.hypotheses)) * cost
	cost.name = "Unregularized cost"
	network.cost = cost
	logger.info(network.cost)
	
	return network

