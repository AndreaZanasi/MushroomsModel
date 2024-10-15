import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
number_of_classes = 9
model.fc = nn.Linear(num_ftrs, number_of_classes)
model.load_state_dict(torch.load('weights\model_weights(lr=0.001)).pth'))
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3914, 0.3697, 0.2815], std=[0.2291, 0.2094, 0.2031])
])

def predict_image(image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

mushrooms_dir = r'Mushrooms'
class_names = sorted(os.listdir(mushrooms_dir))

image_paths = [r'path/to/Agaricus',
               r'path/to/Amanita',
               r'path/to/Boletus',
               r'path/to/Cortinarius',
               r'path/to/Entoloma',
               r'path/to/Hygrocybe',
               r'path/to/Lactarius',
               r'path/to/Russula',
               r'path/to/Suillus']

for image_path in image_paths:
    actual_class = class_names[image_paths.index(image_path)]
    predicted_class = predict_image(image_path, class_names)
    print(f"Actual: {actual_class}, Predicted: {predicted_class}")
