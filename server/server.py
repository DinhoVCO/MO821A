import flwr as fl
import os
from typing import Dict, List, Tuple
from flwr.common import FitRes, Parameters, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
import time

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, log_file: str = "metrics.log"):
        super().__init__()
        self.log_file = log_file
        self.round_start_time = 0
        self.upstream_start_time = 0
        self.upstream_end_time = 0
        self.computation_time = 0

        #Create the log file and write the header
        with open(self.log_file, "w") as f:
            f.write("Round, Loss, Accuracy, F1, Computation Time, Communication Time Upstream, Communication Time Downstream\n")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
        self.round_start_time = time.time()
        self.upstream_start_time = None  # Reset at the start of each round
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        if self.upstream_start_time is None:
            self.upstream_start_time = time.time()  # Capture the time when the first client sends parameters
        #self.computation_start_time = time.time()
        aggregated_params, fit_metrics = super().aggregate_fit(server_round, results, failures)
        self.upstream_end_time = time.time()

        # Collect computation times from all clients
        computation_times = [res.metrics["computation_time"] for _, res in results]
        mean_computation_time = sum(computation_times) / len(computation_times)
        self.computation_time = mean_computation_time

        # Add mean computation time to fit metrics
        fit_metrics["mean_computation_time"] = mean_computation_time
        return aggregated_params, fit_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, Scalar]]:
        if not results:
            return float("inf"), {}
        
        # Aggregate loss, accuracy and f1 score
        loss_aggregated = sum([res.metrics["loss"] for _, res in results]) / len(results)
        accuracy_aggregated = sum([res.metrics["accuracy"] for _, res in results]) / len(results)
        f1_aggregated = sum([res.metrics["f1"] for _, res in results]) / len(results)

        communication_time_upstream = self.upstream_end_time - self.upstream_start_time
        communication_time_downstream = self.round_start_time - self.round_start_time
        
        #Log the metrics
        with open(self.log_file, "a") as f:
            f.write(f"{server_round}, {loss_aggregated}, {accuracy_aggregated}, {f1_aggregated}, {self.computation_time}, {communication_time_upstream}, {communication_time_downstream}\n")

        print(f"Round {server_round} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}, F1: {f1_aggregated}")
        print(f"Computation Time: {self.computation_time}, Communication Time Upstream: {communication_time_upstream}, Communication Time Downstream: {communication_time_downstream}")
        
        return loss_aggregated, {"accuracy": accuracy_aggregated, "f1": f1_aggregated}

# Inicia el servidor
if __name__ == "__main__":
    strategy = CustomStrategy(log_file="metrics.log")
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
    #fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3)) #si


