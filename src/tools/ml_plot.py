import matplotlib.pyplot as plt
from typing import List, Dict, Any
from core.ml_cost import calculate_logistic_cost
from dto.ml_matrix_base import Matrix
from dto.ml_predictions import Predictions
from tools.logger import logger

class Plot:
	
	all_costs: List[float]
	_costs_per_feature: Dict[str, List[float]]
	
	def __init__(self):
		self.all_costs: List[List[float]] = []
		self._costs_per_feature = {}
		
	def calculate_cost(self, predictions: Predictions, y_actual: Matrix) -> None:
		cost: Matrix = calculate_logistic_cost(predictions, y_actual)
		self.all_costs.extend(cost.points)
		print(f"[theta_transpose_x] cost = J(theta) = {cost}")
		
	def add(self, costs: Matrix):
		self.all_costs.extend(costs.points)
	
	def show(self):
		axes = plt.gca()
		x_labels = []
		print(f"plotting cost ...")
		for i in range(0, len(self.all_costs)):
			x_labels.append(i)
		plt.plot(x_labels, self.all_costs, 'o')
		plt.show()
	
	def cast_cost_to_payload(self) -> List[Dict[str, Any]]:
		logger.info("cast_cost_to_payload :: Casting costs to payload ...")
		n_features: int = len(self.all_costs[0]) if len(self.all_costs) > 0 else 0
		costs_per_feature: Dict[str, List[float]] = {}

		for i_feature in range(0, n_features):
			costs_per_feature[str(i_feature)] = []
		
		for cost_entry in self.all_costs:
			for index, cost in enumerate(cost_entry):
				costs_per_feature[str(index)].append(cost)
		
		cost_payload: List[Dict[str, Any]] = []
		for feature, costs in costs_per_feature.items():
			cost_payload.append(
				{
					'type': 'line',
					'data': costs,
					'label': feature,
					'showMark': False
				}
			)
		logger.info("cast_cost_to_payload :: complete")
		return cost_payload
	