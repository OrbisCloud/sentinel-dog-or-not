from typing import List
from math import log
from dto.ml_matrix_base import Matrix, transpose_
from dto.ml_hypothesis import Hypothesis
from dto.ml_theta import Theta


def log_cost(y_predicted: float, y_actual: float) -> float:
	cost: float = (-y_actual * log(y_predicted)) - (1 - y_actual) * log(1 - y_predicted)
	return cost


def squared_cost(y_predicted: float, y_actual: float) -> float:
	return (1 / 2) * pow((y_actual - y_predicted), 2)
	

def calculate_logistic_cost(y_predicted: Matrix, y_actual: Matrix) -> Matrix:	
	log_cost: Matrix = ((-1 * y_actual) ** y_predicted.apply(log)) - ((1 - y_actual) ** (1 - y_predicted).apply(log))		
	log_cost: Matrix = (1 / log_cost.width) * log_cost.squash()
	log_cost.name = "logistic cost"
	return log_cost


def calculate_quadratic_cost(y_predicted: Matrix, y_actual: Matrix) -> Matrix:
	squared_cost: Matrix = (y_actual - y_predicted) ** (y_actual - y_predicted)
	av_cost: Matrix = (1 / (2 * squared_cost.width)) * squared_cost
	return av_cost

