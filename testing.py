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
model.load_state_dict(torch.load('weights\model_weights(lr=0.01,mom=0.9,wd=0.003,pretr=None,bs=32,ep=50,size=300,trainacc=99.80637473935062,testacc=100.0).pth'))
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0004, 0.0003, 0.0002], std=[1.0118, 1.0145, 1.0147])
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

for image_path in image_paths:
    actual_class = class_names[image_paths.index(image_path)]
    predicted_class = predict_image(image_path, class_names)
    print(f"Actual: {actual_class}, Predicted: {predicted_class}")
