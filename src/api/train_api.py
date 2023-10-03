from flask import Blueprint
from dto.ml_hypothesis import Hypothesis
from dto.ml_network import Network
from core.ml_propagation import propagate
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from tools.logger import logger
from dto.ml_hypothesis import Hypothesis
from tools.ml_load import import_csv_file
from tools.ml_plot import Plot
from typing import List, Dict, Any
import pickle
from static.vars import pickle_file, path
from flask import Response


train_api: Blueprint = Blueprint(name='train_api', import_name='train_api')


@train_api.after_request
def add_corse_header_train_api(response: Response) -> Response:
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    return response


def pickle_network(network: Network) -> None:
    stream = pickle_file.open('wb')
    pickle.dump(network, stream)
    stream.close()


@train_api.route('/backprop/<int:epochs>', methods=['GET'])
def run_backpropagation_algorithm(epochs: int) -> List[Dict[str, Any]]:
    logger.info('run_backpropagation_algorithm :: epochs :: %s' % epochs)
    
    hypothesis: Hypothesis = import_csv_file(path, standardize=True)
    dog_or_not: Network = Network(hypothesis, layers=3, alpha=0.5)
    plot: Plot = Plot()

    for _ in range(0, epochs):
        dog_or_not = propagate(dog_or_not, metrics=False)
        plot.add(dog_or_not.cost)
    pickle_network(dog_or_not)
    return plot.cast_cost_to_payload()


# @train_api.route('/gradientdescent/<int:epochs>', methods=['GET'])
# def run_gradient_descent_algorithm(epochs: int):
#     hypothesis: Hypothesis = import_csv_file(path)
#     plot: Plot = Plot()

#     theta: Theta = Theta(4, 5, 0.2)
#     predictions: Matrix
#     for _ in range(1, epochs):
#         predictions: Matrix = calculate_predictions(theta, hypothesis, metrics=False)
#         plot.calculate_cost(predictions, hypothesis.mat_y())
#         theta: Matrix = calculate_gradient_descent(hypothesis, theta, predictions, metrics=False)
#     return plot.all_costs
