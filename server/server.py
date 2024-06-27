import flwr as fl
import os


# Inicia el servidor
if __name__ == "__main__":
    #fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=3))
    fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3))

#fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3)) #si
#fl.server.start_server(server_address=os.environ['SERVER_IP'],config=fl.server.ServerConfig(num_rounds=3)) #no
#fl.server.start_server(server_address="server:8080",config=fl.server.ServerConfig(num_rounds=3)) #no
#fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=3)) #si
#fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))  #si