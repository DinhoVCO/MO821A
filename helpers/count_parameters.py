import torch
import torchvision.models as tmodels
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self) -> None:
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Crear instancias de los modelos
resnet18 = tmodels.resnet18()
mobilenetv2 = tmodels.mobilenet_v2()
net1 = Net1()
# Función para contar parámetros
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Contar y imprimir la cantidad de parámetros
resnet18_params = count_parameters(resnet18)
mobilenetv2_params = count_parameters(mobilenetv2)
net1_params = count_parameters(net1)

print(f'ResNet-18 tiene {resnet18_params / 1e6:.2f} millones de parámetros.')
print(f'MobileNetV2 tiene {mobilenetv2_params / 1e6:.2f} millones de parámetros.')
print(f'Net1 tiene {net1_params / 1e6:.2f} millones de parámetros.')
