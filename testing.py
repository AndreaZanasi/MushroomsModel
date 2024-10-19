import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from colorama import Fore, Style
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
number_of_classes = 9
model.fc = nn.Linear(num_ftrs, number_of_classes)
model.load_state_dict(torch.load(r'weights\model_weights(lr=0.01,mom=0.9,wd=0.003,pretr=Yes,bs=32,ep=50,size=350,trainacc=98.07864164432529,testacc=98.8978254393804).pth'))
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize([250, 250]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2787, 0.2223, 0.1592], std=[0.2433, 0.2235, 0.2131])
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

image_paths = [r'c:\Users\Utente\Desktop\Agaricus_campestris(fs-03).jpg',
               r'c:\Users\Utente\Desktop\amanita-muscaria10.jpg',
               r'c:\Users\Utente\Desktop\boletus-reticulatus1.jpg',
               r'c:\Users\Utente\Desktop\Cortinarius_xanthodryophilus_db-01.jpg',
               r'c:\Users\Utente\Desktop\entoloma-caeruleum1.jpg',
               r'c:\Users\Utente\Desktop\Hygrocybe_coccinea(mgw-04).jpg',
               r'c:\Users\Utente\Desktop\Lactarius-vellereus-11-X-2007-066.jpg',
               r'c:\Users\Utente\Desktop\russula-seven.jpg',
               r'c:\Users\Utente\Desktop\Suillus_granulatus.jpg']

correct_predictions = 0

for image_path in image_paths:
    actual_class = class_names[image_paths.index(image_path)]
    predicted_class = predict_image(image_path, class_names)
    
    if actual_class == predicted_class:
        print(f"{Fore.GREEN}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")
        correct_predictions += 1
    else:
        print(f"{Fore.RED}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")

total_images = len(image_paths)
accuracy = correct_predictions / total_images
print(f"Accuracy: {correct_predictions}/{total_images} ({accuracy:.2%})")
