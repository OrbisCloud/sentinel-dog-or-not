from math import e
from typing import List


def calculate_sigmoid(z: float) -> float:
	res = 1 / (1 + (pow(e, -z)))
	return res


def calculate_sigmoids(Z: List[float]) -> List[float]:
	ress: List[float] = []
	for z in Z:
		ress.append(calculate_sigmoid(z))
	return ress


