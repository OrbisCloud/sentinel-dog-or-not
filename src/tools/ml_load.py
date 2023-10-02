from io import TextIOWrapper
from typing import List
from dto.ml_hypothesis import Hypothesis
from standard.ml_feature_scaling import standardize_values
from tools.logger import logger
from pathlib import Path


def data_not_empty_and_has_same_length(buffer: TextIOWrapper) -> bool:
	length: int = -1
	for line in buffer:
		if length == -1:
			length = len(line)
		if len(line) != length:
			return False
		return False if length <= 1 else True


def split_x_from_y(path: str) -> (List[List[float]], List[float]):
	file: TextIOWrapper = open(path, 'r')
	x_values: List[List[float]] = []
	y_values: List[float] = []

	if not data_not_empty_and_has_same_length(file):
		print(
			f"[import_csv_file] \nFile must contain at least 2 values per training example."
		)
		print(f"Every training example must be of the same length.")
		raise Exception()
	file.seek(0)

	for line in file:
		values: List[float] = line.replace('\n', '').replace(' ', '').split(',')
		try:
			values = list(map(lambda val: float(val), values))
			x_values.append(values[:len(values) - 1])
			y_values.append(values[len(values) - 1])
		except Exception as e:
			print(f"[import_csv_file] Could not cast '{values}' to float.")
	return (x_values, y_values)
	

def create_one_hot_vector(i: int, max: int) -> List[float]:
	m_vector: List[float] = [1 if i == j+1 else 0 for j in range(0, max)]
	return m_vector
	

def import_csv_file(path: Path, standardize: bool = True) -> Hypothesis:
	logger.info('import_csv_file :: from path: %s' % path.as_posix())
	logger.info('import_csv_file :: loading ...')

	x_values, y_values = split_x_from_y(path.as_posix())
	
	y_max: int = int(max(y_values))
	sigmoid_y_values: List[List[float]] = []
	
	for i in range(0, len(y_values)):
		y_values[i] = create_one_hot_vector(y_values[i], y_max)
	hypothesis: Hypothesis = Hypothesis(x_values, y_values)
	if standardize:
		return Hypothesis(standardize_values(hypothesis).points, y_values)
	else:
		return hypothesis







