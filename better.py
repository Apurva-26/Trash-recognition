import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
import numpy as np

from torchvision.models import ResNet18_Weights



model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


dataset_path = r"C:\Users\apurv\Downloads\archive real\garbage_classification"


train_transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root=dataset_path,)



train_size = int(0.7 * len(dataset))
test_size = int(0.15 * len(dataset))
val_size = len(dataset) - train_size - test_size
seed = 42
generator = torch.Generator().manual_seed(seed)

train_dataset_raw, test_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
    dataset, [train_size, test_size, val_size], generator=generator
)
# train_dataset_raw, test_dataset_raw, val_dataset_raw = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])


train_dataset_raw.dataset.transform = train_transform
test_dataset_raw.dataset.transform = test_transform
val_dataset_raw.dataset.transform = test_transform 




train_loader = DataLoader(train_dataset_raw, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset_raw, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset_raw, batch_size=32, shuffle=True)



class_names = dataset.classes
print(f'Classes: {class_names}')

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))


try:
    model.load_state_dict(torch.load("better_classifier.pth"))
    print("Loaded saved model. Continuing training...")
except:
    print("No saved model found. Starting fresh.")

model.train()

model = model.to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # only training final layer


epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

print("Training complete.")
torch.save(model.state_dict(), "better_classifier.pth")
print("Model saved as better_classifier.pth")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')


def imshow(img, title):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img.cpu() * std + mean  
    npimg = img.numpy().transpose((1, 2, 0))
    npimg = np.clip(npimg, 0, 1) 
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.show()

dataiter = iter(val_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

for i in range(10):
    imshow(images[i].cpu(), f'Pred: {class_names[preds[i]]}, Actual: {class_names[labels[i]]}')

