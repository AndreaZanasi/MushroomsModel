import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from colorama import Fore, Style
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
number_of_classes = 9
model.fc = nn.Linear(num_ftrs, number_of_classes)

model.load_state_dict(torch.load(
    r'weights\model_weights(lr=0.01,mom=0.9,wd=0.003,pretr=Yes,bs=32,ep=50,size=400,trainacc=97.58713136729223,testacc=99.38933571641347).pth',
    map_location=torch.device('cpu'), 
    weights_only=True
))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize([400, 400]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4639, 0.6601, 0.5745], std=[0.9625, 0.9623, 0.9794])
])

def predict_image(image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

script_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(script_dir, 'Test')
if not os.path.exists(test_data_dir):
    raise FileNotFoundError(f"Test data directory '{test_data_dir}' does not exist.")

class_names = sorted(os.listdir(test_data_dir))

test_data = {}
for class_name in class_names:
    class_dir = os.path.join(test_data_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, file_name)
            test_data[image_path] = class_name

correct_predictions = 0
total_predictions = 0

for image_path, actual_class in test_data.items():
    predicted_class = predict_image(image_path, class_names)
    
    if actual_class == predicted_class:
        print(f"{Fore.GREEN}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")
        correct_predictions += 1
    else:
        print(f"{Fore.RED}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")
    
    total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})")