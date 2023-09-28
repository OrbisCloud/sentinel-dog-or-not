from flask import Blueprint
from dto.ml_theta import Theta
from dto.ml_hypothesis import Hypothesis
from dto.ml_network import Network
from core.ml_propagation import propagate
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from tools.logger import logger
from pathlib import Path
from dto.ml_matrix_base import Matrix
from dto.ml_hypothesis import Hypothesis
from core.ml_gradient import calculate_predictions, calculate_gradient_descent
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from typing import List, Dict, Any
from flask.wrappers import Response


path: Path = Path(Path('.').absolute()) / 'src' / 'data' / 'neural_training_data.csv'
run_api: Blueprint = Blueprint(name='run_api', import_name='run_api')


@run_api.after_request
def add_corse_header(response: Response) -> Response:
    logger.info("%20s :: %20s" % ("Response is of type" , type(response)))
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    return response




@run_api.route('/ping', methods=['GET'])
def ping():
    return 'pong'


@run_api.route('/backprop/<int:epochs>', methods=['GET'])
def run_backpropagation_algorithm(epochs: int) -> List[Dict[str, Any]]:
    logger.info('run_backpropagation_algorithm :: epochs :: %s' % epochs)
    
    hypothesis: Hypothesis = import_csv_file(path, standardize=True)
    dog_or_not: Network = Network(hypothesis, layers=3, alpha=1)
    plot: Plot = Plot()

    for _ in range(0, epochs):
        dog_or_not = propagate(dog_or_not, metrics=False)
        plot.add(dog_or_not.cost)
    return plot.cast_cost_to_payload()


@run_api.route('/gradientdescent/<int:epochs>', methods=['GET'])
def run_gradient_descent_algorithm(epochs: int):
    hypothesis: Hypothesis = import_csv_file(path)
    plot: Plot = Plot()

    theta: Theta = Theta(4, 5, 0.2)
    predictions: Matrix
    for _ in range(1, epochs):
        predictions: Matrix = calculate_predictions(theta, hypothesis, metrics=False)
        plot.calculate_cost(predictions, hypothesis.mat_y())
        theta: Matrix = calculate_gradient_descent(hypothesis, theta, predictions, metrics=False)
    return plot.all_costs
