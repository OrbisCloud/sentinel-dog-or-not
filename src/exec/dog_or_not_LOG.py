from dto.ml_matrix_base import Matrix
from dto.ml_hypothesis import Hypothesis
from dto.ml_theta import Theta
from core.ml_gradient import calculate_predictions, calculate_gradient_descent
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot



hypothesis: Hypothesis = import_csv_file("data/neural_training_data.csv")
theta: Theta = Theta(4, 5, 0.2)
plot: Plot = Plot()


epochs: int = 1500

predictions: Matrix
for i in range(1, epochs):
	predictions: Matrix = calculate_predictions(theta, hypothesis, metrics=False)
	plot.calculate_cost(predictions, hypothesis.mat_y())
	theta: Matrix = calculate_gradient_descent(hypothesis, theta, predictions, metrics=False)

plot.show()
print(predictions)

