import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import random

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class_names = ['Agaricus', 'Amanita', 'Boletus',
               'Cortinarius', 'Entoloma', 'Hygrocybe',
               'Lactarius', 'Russula', 'Suillus']


# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 87 * 67, 512)  # Adjust the input size based on the output of the conv layers
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 9)  # 9 output classes

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 87 * 67)  # Adjust the size based on the output of the conv layers
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Apply the final output layer
        x = self.fc3(x)
        return x

# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize((700, 541)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder(
    root='/home/zanna/Desktop/Projects/MushroomsModel/Mushrooms',
    transform=transform
)

test_data = torchvision.datasets.ImageFolder(
    root='/home/zanna/Desktop/Projects/MushroomsModel/Mushrooms',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=True, num_workers=2
)
# Instantiate the model, define the loss function and the optimizer
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Starting epoch {epoch}")
    running_loss = 0.0
    image_counter = 0
    for images, labels in train_loader:
        print(f"Image {image_counter}")
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        image_counter +=1
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'mushroom_model.pth')