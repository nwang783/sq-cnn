import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# define dirs
data_dir = "data2"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

all_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=all_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=all_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# print results
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Class names: {train_dataset.classes}")

# load pre-trained model
model = models.resnet18(pretrained=True)

# replace final later for bianry classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# move model to device
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    running_loss = 0.0

    # training step
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameters grads
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Training Loss: {running_loss / len(train_loader)}")

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss - criterion(outputs, labels)
            val_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation loss: {val_loss / len(val_loader)}")
    print(f"Validation accuracy: {100 * correct / total:.2f}%")

print("Traing complete!")

torch.save(model.state_dict(), "squirrel_classifier_v3.pth")
