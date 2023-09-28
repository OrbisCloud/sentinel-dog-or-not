from typing import List
from dto.ml_matrix_base import Matrix


# Variance, sum of squared difference between values and the average value
def variance(values: List[float]) -> float:
    m: float = mean(values)
    return sum([(x-m)**2 for x in values])


def coveriance(x: List[float], y: List[float]) -> float:
    covar: float = 0
    x_mean: float = mean(x)
    y_mean: float = mean(y)
    for i in range(0, len(x)):
    [covar = covar + (x[i] - x_mean) * (y[i] - y_mean) for i in range(0, len(x))]
    return covar

