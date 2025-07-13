from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from functools import reduce
from utils_quantisation import quantize, dequantize_mean, add_mod, dequantize, CR, N
from run_config import QUANTISATION, edge_weights
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy
from flwr.common import (
    FitIns,
    FitRes,
    Scalar,
)


'''def aggregate_weighted_average(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    total_weighted_examples = sum(num_examples * edge_weight for (_, num_examples), edge_weight in zip(results, edge_weights))

    for (weights, num_examples) in results:
        if np.isnan(weights).any() or np.isinf(weights).any():
            raise ValueError("Weights contain NaN or Inf values.")
        if num_examples < 0:
            raise ValueError("Number of examples cannot be negative.")

    # Ensure edge_weights is a NumPy array
    edge_weights_array = np.array(edge_weights, dtype=np.float64)

    # Check for shape compatibility
    if len(results) != len(edge_weights_array):
        raise ValueError("The length of results and edge_weights must match.")

    # Calculate weighted weights with broadcasting
    weighted_weights = [
        [np.clip(layer.astype(np.float64) * (num_examples * edge_weights_array[i]), -1e10, 1e10) for layer in weights]
        for i, (weights, num_examples) in enumerate(results)
    ]

    if total_weighted_examples <= 0:
        raise ValueError("Total weighted examples must be greater than zero.")

    # Compute the weighted average of the model parameters
    weights_avg: NDArrays = [
        (reduce(np.add, layer_updates) / np.float64(total_weighted_examples)).astype(np.float64)
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_avg
'''
def aggregate_weighted_average(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """
    Compute the weighted average of model parameters using edge weights defined in run_config.py.

    Parameters:
        results (List[Tuple[NDArrays, int]]): A list of tuples containing model parameters (NDArrays) and the number of examples (int).

    Returns:
        NDArrays: The weighted average of model parameters.
    """
    # Compute total weighted examples
    total_weighted_examples = sum(num_examples * edge_weight for (_, num_examples), edge_weight in zip(results, edge_weights))

    # Check for NaN or Inf values and validate num_examples
    for (weights, num_examples) in results:
        if np.isnan(weights).any() or np.isinf(weights).any():
            raise ValueError("Weights contain NaN or Inf values.")
        if num_examples < 0:
            raise ValueError("Number of examples cannot be negative.")

    # Calculate weighted weights with clipping
    max_value = 1e10  # Define a maximum threshold
    weighted_weights = [
        [np.clip(layer.astype(np.float64) * np.float64(num_examples * edge_weight), -max_value, max_value) for layer in weights]
        for (weights, num_examples), edge_weight in zip(results, edge_weights)
    ]

    # Ensure total_weighted_examples is valid
    if total_weighted_examples <= 0:
        raise ValueError("Total weighted examples must be greater than zero.")

    # Compute the weighted average of the model parameters
    weights_avg: NDArrays = [
        (reduce(np.add, layer_updates) / np.float64(total_weighted_examples)).astype(np.float64)
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_avg

def aggregate_qt(results: List[Tuple[NDArrays, np.int64]]) -> NDArrays:
    """
    Compute the weighted average of quantized model parameters and dequantize the result using edge weights defined in run_config.py.

    Parameters:
        results (List[Tuple[NDArrays, int]]): A list of tuples containing quantized model parameters (NDArrays) and the number of examples (int).

    Returns:
        NDArrays: The weighted and dequantized average of model parameters.
    """
    total_weighted_examples = sum(num_examples * edge_weight for (_, num_examples), edge_weight in zip(results, edge_weights))
    print('Total weighted examples:')
    print(total_weighted_examples)

    # Multiply each client's quantized weights by the number of examples and edge weight
    weighted_weights = [
        [layer * (num_examples * edge_weight) for layer in weights]
        for (weights, num_examples), edge_weight in zip(results, edge_weights)
    ]

    # Check for excessively large values
    max_value = 1e10  # Define a maximum threshold
    for layer in weighted_weights:
        for weight in layer:
            if np.abs(weight).max() > max_value:
                print("Warning: Exceedingly large values detected in weighted contributions.")

    # Compute average weights of each layer (still quantized)
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    # Dequantize the aggregated weights
    print('Dequantizing...')
    weights_deq = dequantize(weights_prime, CR, N, total_weighted_examples)

    # Normalize the dequantized weights by the total weighted examples
    if total_weighted_examples <= 0:
        raise ValueError("Total weighted examples must be greater than zero.")
    
    weights_deq_weigh = np.divide(weights_deq, total_weighted_examples)

    print('Final dequantized and weighted model parameters:')
    print(weights_deq_weigh)

    return weights_deq_weigh

def aggregate_qt2(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    total_weighted_examples = sum(num_examples * edge_weight for (_, num_examples), edge_weight in zip(results, edge_weights))
    print('Total weighted examples:')
    print(total_weighted_examples)

    # Multiply each client's quantized weights by the number of examples and edge weight
    weighted_weights = [
        [layer * (num_examples * edge_weight) for layer in weights]
        for (weights, num_examples), edge_weight in zip(results, edge_weights)
    ]

    # Check for excessively large values
    max_value = 1e10  # Define a maximum threshold
    for layer in weighted_weights:
        for weight in layer:
            if np.abs(weight).max() > max_value:
                print("Warning: Exceedingly large values detected in weighted contributions.")

    # Compute average weights of each layer (still quantized)
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    print('num_examples_total', total_weighted_examples)
    weights_deq_weigh = dequantize_mean(weights_prime, CR, N, total_weighted_examples)

    print('weights_deq_weigh', weights_deq_weigh)

    return weights_deq_weigh

class LocalUpdatesStrategy(fl.server.strategy.FedAvg):
    """
    A strategy for federated averaging that considers local updates with optional quantization.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using weighted average with support for quantized parameters.

        Parameters:
            server_round (int): The current round of the server.
            results (List[Tuple[ClientProxy, FitRes]]): The results from clients' local training.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): A list of clients that failed to return results.

        Returns:
            Tuple[Optional[Parameters], Dict[str, float]]: Aggregated parameters and aggregated metrics.
        """

        if not results or (not self.accept_failures and failures):
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]

        if QUANTISATION:
            parameters_aggregated = ndarrays_to_parameters(aggregate_qt2(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(aggregate_weighted_average(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
