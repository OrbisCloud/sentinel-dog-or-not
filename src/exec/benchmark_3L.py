# Inspired by Matt Mazur's blogpost on: 
# 		Partial derivative calculations of back propagation algorithm
# 		Source & credit: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

from dto.ml_hypothesis import Hypothesis
from dto.ml_theta import Theta
from dto.ml_network import Network
from core.ml_propagation import calculate_forward_propagation, calculate_back_propagation, propagate
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from dto.ml_matrix_base import Matrix

hypothesis: Hypothesis = import_csv_file("data/mazur.csv", standardize=False)
mazur: Network = Network(hypothesis, layers=3, alpha=1)
plot: Plot = Plot()


theta_1 = [
	[.15, .20, .35],
	[.25, .30, .35]
	]
theta1: Theta = Theta(None, None, 1, theta_1)

theta_2 = [
	[.40, .45, .60],
	[.50, .55, .60]
	]
theta2: Theta = Theta(None, None, 1, theta_2)

y_actual: Matrix = Matrix([[0.01, 0.99]], "y_actual")


mazur.set_theta(0, theta1)
mazur.set_theta(1, theta2)
mazur.hypotheses[0].y = [[0.01, 0.99]]


epochs: int = 2000
for i in range(0, epochs):
	mazur = propagate(mazur, metrics=False)
	plot.add(mazur.cost)
plot.show()

