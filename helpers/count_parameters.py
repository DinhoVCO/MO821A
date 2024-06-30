import torch
import torchvision.models as models

# Crear instancias de los modelos
resnet18 = models.resnet18()
mobilenetv2 = models.mobilenet_v2()

# Función para contar parámetros
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Contar y imprimir la cantidad de parámetros
resnet18_params = count_parameters(resnet18)
mobilenetv2_params = count_parameters(mobilenetv2)

print(f'ResNet-18 tiene {resnet18_params / 1e6:.2f} millones de parámetros.')
print(f'MobileNetV2 tiene {mobilenetv2_params / 1e6:.2f} millones de parámetros.')
