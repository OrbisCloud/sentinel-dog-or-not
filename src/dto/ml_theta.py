from typing import List
from dto.ml_matrix_base import Matrix
from random import uniform


class Theta(Matrix):
	alpha: float
	lambd: float

	def __init__(self, width: int, height: int, alpha: float, points: List[List[float]] = None):
		if points is not None:
			super().__init__(points, "Theta")
		else:
			epsilon: float = 1e-4
			m_points: List[List[float]] = []
			for i in range(0, width):
				m_points.append([])
				for j in range(0, height):
					# m_points[i].append(uniform(-epsilon, epsilon))
					m_points[i].append(uniform(-1, 1))
			super().__init__(m_points, "Theta")
		self.alpha = alpha
		self.lambd = 0.005
		

def create_empty_theta(t: Theta) -> Theta:
	empty_theta_vals: List[List[float]] = []
	for i in range(0, t.l):
		empty_theta_arr: List[float] = [0] * t.w
		empty_theta_vals.append(empty_theta_arr)
	return Theta(empty_theta_vals)

