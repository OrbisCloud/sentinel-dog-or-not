from typing import List


def min_max_normalization(values: List[List[float]]) -> List[List[float]]:
    all_values = []
    [all_values.extend(value_list) for value_list in values]
    
    minimum: float = float(min(all_values))
    maximum: float = float(max(all_values))
    diff: float = float(maximum - minimum)
    
    for i in range(0, len(values)):
        for j in range(0, len(values[i])):
            print(f"{values[i][j]} -> before")
            values[i][j] = (values[i][j]) / maximum
            print(f"{values[i][j]} -> after")
    
    
    return values
