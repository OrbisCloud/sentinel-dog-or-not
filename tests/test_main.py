"""
Tests for main
"""
import pytest
from src.main import add
from src.logger import logger


@pytest.mark.parametrize(('number_1', 'number_2', 'result'), [
    (1, 2, 3),
    (3, 4, 7),
    (5, 8, 13),
    (9, -2, 7),
    (1, -1, 0)
])
def test_add(number_1: int, number_2: int, result: int) -> None:
    # logger.info(sys.path)
    logger.info("Hello world")
    assert add(number_1, number_2) == result

