from dto.ml_hypothesis import Hypothesis
from dto.ml_network import Network
from core.ml_propagation import propagate
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from typing import List


hypothesis: List[Hypothesis] = import_csv_file("data/neural_training_data.csv", standardize=True)
dog_or_not: Network = Network(hypothesis, layers=3, alpha=1)
plot: Plot = Plot()

epochs: int = 100
for i in range(0, epochs):
	dog_or_not = propagate(dog_or_not, metrics=False)
	plot.add(dog_or_not.cost)
plot.show()

