import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes):
    model = models.resnet50(pretrained=True)
    
    # Freeze pre-trained layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the fully connected layer for custom classification task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
