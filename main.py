import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 87 * 67, 512)  # Adjust the size based on the output of the conv layers
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # Assuming 9 classes for the mushrooms dataset

    def forward(self, x):
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

def main():
    # Define transformations for the input data
    transform = transforms.Compose([
        transforms.Resize((700, 541)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.ImageFolder(
        root='Mushrooms',
        transform=transform
    )

    test_data = torchvision.datasets.ImageFolder(
        root='Mushrooms',
        transform=transform
    )

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_data, batch_size=32, shuffle=True, num_workers=2
    )

    model = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        running_loss = 0.0
        image_counter = 0
        t = time.time()  # Initialize time variable

        for images, labels in train_loader:
            if image_counter % 10 == 0 and image_counter != 0:
                print(f"Image {image_counter}")
                print(f"Time per 10 images: {time.time() - t}")
                t = time.time()
            image_counter += 1

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
    print('Finished Training')

    # Save the model
    torch.save(model.state_dict(), 'mushroom_model.pth')

if __name__ == '__main__':
    main()