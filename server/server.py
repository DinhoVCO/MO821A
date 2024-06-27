import flwr as fl
import os
from typing import Dict, List, Tuple


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, log_file: str = "metrics.log"):
        super().__init__()
        self.log_file = log_file
        #Create the log file and write the header
        with open(self.log_file, "w") as f:
            f.write("Round, Loss, Accuracy\n")

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        if not results:
            return float("inf"), {}
        
        # Aggregate loss and accuracy
        loss_aggregated = sum([fit_res.metrics["loss"] for _, fit_res in results]) / len(results)
        accuracy_aggregated = sum([fit_res.metrics["accuracy"] for _, fit_res in results]) / len(results)
        
        #Log the metrics
        with open(self.log_file, "a") as f:
            f.write(f"{rnd}, {loss_aggregated}, {accuracy_aggregated}\n")
            
        print(f"Round {rnd} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}")
        
        return loss_aggregated, {"accuracy": accuracy_aggregated}

# Inicia el servidor
if __name__ == "__main__":
    strategy = CustomStrategy()
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
    #fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3))

#fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3)) #si
#fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3)) #no
#fl.server.start_server(server_address="server:8080",config=fl.server.ServerConfig(num_rounds=3)) #no
#fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=3)) #si
#fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))  #si