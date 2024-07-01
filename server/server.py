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
        self.upstream_start_time = 0
        self.upstream_end_time = 0
        self.computation_start_time = 0
        self.computation_end_time = 0

        #Create the log file and write the header
        with open(self.log_file, "w") as f:
            f.write("Round, Loss, Accuracy, F1, Computation Time, Communication Time Upstream, Total Time\n")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        self.upstream_end_time = time.time()
        
        # Collect min start trainin time and max end training time
        start_times = [res.metrics["start_time"] for _, res in results]
        end_times = [res.metrics["end_time"] for _, res in results]
        self.computation_start_time = min(start_times)
        self.computation_end_time = max(end_times)
        self.upstream_start_time = self.computation_end_time

        # Add mean computation time to fit metrics
        self.computation_time = self.computation_end_time - self.computation_start_time
        aggregated_params, fit_metrics = super().aggregate_fit(server_round, results, failures)
        fit_metrics["computation time"] = self.computation_time
        
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

        #compute communication and total time
        communication_time_upstream = self.upstream_end_time - self.upstream_start_time
        total_time = self.upstream_end_time - self.computation_start_time

        #Log the metrics
        with open(self.log_file, "a") as f:
            f.write(f"{server_round}, {loss_aggregated}, {accuracy_aggregated}, {f1_aggregated}, {self.computation_time}, {communication_time_upstream}, {total_time}\n")

        print(f"Round {server_round} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}, F1: {f1_aggregated}")
        print(f"Computation Time: {self.computation_time}, Communication Time Upstream: {communication_time_upstream}, Total Time: {total_time}")
        
        return loss_aggregated, {"accuracy": accuracy_aggregated, "f1": f1_aggregated}

# Inicia el servidor
if __name__ == "__main__":
    strategy = CustomStrategy(log_file="metrics.log")
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
    #fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3)) #si


