import flwr as fl
import os
from typing import Dict, List, Tuple

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, log_file: str = "metrics.log"):
        super().__init__()
        self.log_file = log_file
        #Create the log file and write the header
        with open(self.log_file, "w") as f:
            f.write("Round, Loss, Accuracy, F1\n")

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        if not results:
            return float("inf"), {}
        
        # Aggregate loss, accuracy and f1 score
        loss_aggregated = sum([res.metrics["loss"] for _, res in results]) / len(results)
        accuracy_aggregated = sum([res.metrics["accuracy"] for _, res in results]) / len(results)
        f1_aggregated = sum([res.metrics["f1"] for _, res in results]) / len(results)
        
        #Log the metrics
        with open(self.log_file, "a") as f:
            f.write(f"{rnd}, {loss_aggregated}, {accuracy_aggregated}, {f1_aggregated}\n")

        print(f"Round {rnd} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}, F1: {f1_aggregated}")
        
        return loss_aggregated, {"accuracy": accuracy_aggregated, "f1": f1_aggregated}

# Inicia el servidor
if __name__ == "__main__":
    strategy = CustomStrategy(log_file="metrics.log")
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
    #fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))
    #fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3)) #si
