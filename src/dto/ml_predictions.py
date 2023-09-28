from typing import List
from dto.ml_matrix_base import Matrix


class Predictions(Matrix):
	def __init__(self, _P: List[List[float]]):
		super().__init__(_P, "Predictions")

