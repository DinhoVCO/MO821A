# MO821A - Tópicos em Redes de Computadores II
## Análise de desempenho do aprendizado federado: uma abordagem de simulac ¸˜ ao e emulação em  dispositivos heterogêneos

## Quick start
### Building images
- CLIENT (directory : client folder) : 
`docker build -t fl-server .`

- SERVER (directory : server folder) : 
`docker build -t fl-client .`

### Run the containers
In the base directory:
`docker compose -f docker-compose.yml --compatibility up` or `docker-compose up -d` 

### Personalizar variables de entorno
* modificar el arquivo **.env** e executar `docker compose -f docker-compose.yml --compatibility up` 

O :
* `MODEL=CNN DATASET=MNISTS docker-compose up -d` 

## Base neural network models:
    - ResNet-18 (11689512 parameters)
    - MobileNetV2 (3504872 parameters)
    - Personalized1(Ours)

## Base neural network models:
    - MNIST
    - CIFAR10
