import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import time
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
            return img.convert('RGB')  # Convert image to RGB
        except OSError as e:
            print(f"Error loading image at {path}: {e}")
            return None
torchvision.datasets.folder.pil_loader = pil_loader

# To use GPU for parallel computing
def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, epochs):
    device = set_device()

    for epoch in range(epochs):
        start_time = time.time()
        print(f'Starting epoch: {epoch + 1}')
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_accuracy = 100.00 * running_correct /  total

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f'-Training Data: Got {running_correct} out of {total} images. Accuracy: {epoch_accuracy}% Loss: {epoch_loss}')
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')
        evaluate_model_on_test_set(model, test_loader)

    print("Finished")
    return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    start_time = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    
    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    end_time = time.time()
    evaluation_duration = end_time - start_time
    print(f'-Testing Data:  Got {predicted_correctly_on_epoch} out of {total} images. Accuracy: {epoch_acc}%')
    print(f'Evaluation completed in {evaluation_duration:.2f} seconds')


mean = [0.3914, 0.3697, 0.2815]
std = [0.2291, 0.2094, 0.2031]

train_transforms = transforms.Compose([
    # Size of images afflicts the performance of the model
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    #transforms.RandomHorizontalFlip(), to random flip images to train on different types
])
#Batch Size afflicts the performance of the model
train_dataset = torchvision.datasets.ImageFolder(root='Mushrooms', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
test_dataset = torchvision.datasets.ImageFolder(root='Mushrooms', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


model = models.resnet18(weights=None)           # Resnet18 model weights=None for not already trained
num_ftrs = model.fc.in_features                     # Size of each input sample
number_of_classes = 9
model.fc = nn.Linear(num_ftrs, number_of_classes)   # Prepare the matrices for forward propagation
device = set_device()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()                     # Useful function for classification problems
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.003) #lr most important parameter 

train_nn(model, train_loader, test_loader, loss_fn, optimizer, 100)

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')