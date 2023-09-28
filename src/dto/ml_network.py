from typing import List
from copy import deepcopy
from dto.ml_matrix_base import Matrix, transpose_, zero_matrix_, mat_
from dto.ml_theta import Theta
from dto.ml_hypothesis import Hypothesis
from dto.ml_predictions import Predictions


class Network:
	layers: int
	alpha: float
	
	master_hypothesis: Hypothesis
	hypotheses: List[Hypothesis]
	
	thetas: List[Theta]
	predictions: List[Predictions]
	deltas: List[Matrix]
	
	y_actual: Matrix
	cost: Matrix
	
	def __init__(self, hypothesis: Hypothesis, layers: int, alpha: float=0.25):
		
		self.layers = layers
		self.alpha = alpha
		self.thetas = []
		self.predictions = []
		self.pre_sigmoids = []
		self.deltas = []
		self.hypotheses = []
		
		# Create seperate hypothesis for every training example
		self.master_hypothesis = hypothesis
		for m in range(0, hypothesis.width):
			training_example: List[List[float]] = [hypothesis.points[m]]
			actual_y: List[List[float]] = [hypothesis.mat_y().points[m]]
			self.hypotheses.append(Hypothesis(training_example, actual_y))

		# Populating the theta, delta and prediction layers
		for l in range(0, layers):
			theta: Theta = Theta(hypothesis.height, hypothesis.height+1, alpha)						# +1, for bias node
			delta: Matrix = zero_matrix_(hypothesis.height, hypothesis.height)
			predictions: Matrix = zero_matrix_(1, 1)
			pre_sigmoid: Matrix = zero_matrix_(1, 1)
			
			theta.name = f"layer {l}: {theta.name}"
			delta.name = f"layer {l}: delta"
			predictions.name = f"layer {l}: predictions"
			pre_sigmoid.name = f"layer {l}: pre-sigmoid"
			
			self.thetas.append(theta)
			self.deltas.append(delta)
			self.predictions.append(predictions)
			self.pre_sigmoids.append(pre_sigmoid)
		
		self.set_training_example(0)
		last_theta:Theta = Theta(self.y_actual.height, hypothesis.height+1, alpha)			# +1, for bias node
		last_delta: Matrix = zero_matrix_(self.y_actual.height, hypothesis.height)
		
		self.set_theta(layers-2, last_theta)
		self.set_delta(layers-2, last_delta)
			
	def set_layer(self, l: int, pred: Predictions):
		pred.name = f"layer {l}: Predictions"
		self.predictions[l] = []
		self.predictions[l] = pred
	
	def set_theta(self, l: int, theta: Theta):
		theta.name = f"layer: {l}: Theta"
		self.thetas[l] = []
		self.thetas[l] = theta
	
	def set_delta(self, l: int, delta: Matrix):
		delta.name = f"layer {l}: Cumulative Delta"
		self.deltas[l] = []
		self.deltas[l] = delta
	
	def set_pre_sigmoid(self, l: int, pre_sigmoid: Matrix):
		pre_sigmoid.name = f"layer {l}: Pre-Sigmoid"
		self.pre_sigmoids[l] = []
		self.pre_sigmoids[l] = pre_sigmoid
		
	def get_layer(self, l: int) -> Predictions:
		return mat_(None, None, self.predictions[l])
	
	def get_theta(self, l: int) -> Theta:
		return mat_(None, None, self.thetas[l])
	
	def get_delta(self, l: int) -> Matrix:
		return mat_(None, None, self.deltas[l])
	
	def get_pre_sigmoid(self, l: int) -> Matrix:
		return mat_(None, None, self.pre_sigmoids[l])
	
	def get_hypothesis(self, training_example: int) -> Hypothesis:
		return mat_(None, None, self.hypotheses[training_example])
		
	def set_training_example(self, training_example: int):
		training_values: Predictions = Predictions(deepcopy(self.hypotheses[training_example].points))
		training_values.name = f"layer 0: training example ({training_example})"
		self.set_layer(0, training_values)
		self.y_actual = self.hypotheses[training_example].mat_y()
			
	def reset_deltas(self):
		for delta in self.deltas:
			for i in range(0, delta.width):
				for j in range(0, delta.height):
					delta.points[i][j] = 0
					

