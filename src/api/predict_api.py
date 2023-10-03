import json
from flask import Blueprint
from dto.ml_hypothesis import Hypothesis
from dto.ml_network import Network
from dto.ml_matrix_base import Matrix
from dto.ml_hypothesis import Hypothesis
from typing import List
from tools.ml_load import standardize_values
from core.ml_propagation import calculate_forward_propagation
from flask import request
import pickle
from static.vars import pickle_file
from flask import Response


predict_api: Blueprint = Blueprint(name='predict_api', import_name='predict_api')


@predict_api.after_request
def add_corse_header_predict_api(response: Response) -> Response:
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Headers'] = 'content-type'
    return response


def unpickle_network() -> Network:
    stream: bytes = pickle_file.open('rb').read()
    network: Network = pickle.loads(stream)
    return network


@predict_api.route('/backprop/predict', methods=['POST'])
def predict() -> List[float]:
    candidate: List[float] = json.loads(request.data)

    # General input validation
    if len(candidate) != 4:
        raise Exception('Candidate should be 4 floats long')
    
    for number in candidate:
        float(number)

    trained_network: Network = unpickle_network()
    
    # Add bias node
    candidate.insert(-1, 1)

    # Construct hypothesis
    x_values: List[List[float]] = [candidate]
    y_values: List[List[float]] = [[0, 0, 0, 0]]


    hypothesis: Hypothesis = Hypothesis(x_values, y_values[0])
    hypothesis = Hypothesis(standardize_values(hypothesis).points, y_values)

    # Add hypothesis to network
    trained_network.add_hypothesis(hypothesis=hypothesis)

    # Set working example to hypothesis and predict by doing a forward pass
    trained_network.set_training_example(-1)

    result: Matrix
    _, __, result = calculate_forward_propagation(trained_network)

    # Print result and hope for the best
    # cost: Matrix = result.apply(log_cost, trained_network.y_actual)
    print("Received result: %s" % result)
    points: List[float] = result.points[0]
    return points