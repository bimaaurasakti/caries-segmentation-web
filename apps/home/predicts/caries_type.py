from PIL import Image
from torchvision import transforms, models

import torch
import torch.nn as nn

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)
    num_classes = 4

    # Add sequential layers
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(torch.load('models/predict-resnext50-bn-asli.pth', map_location=torch.device('cpu')))
    model.eval()

    # Load and preprocess the image
    image = Image.open(image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Load the class labels
    class_labels = ['Dentin','Enamel','Normal','Pulpa']  # Replace with your actual class labels
    
    predicted_label = predicted.item()

    return class_labels[predicted_label]