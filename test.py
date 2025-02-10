import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64*7*7, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,10)
        )

    def forward(self, x):
        return self.network(x) 

device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
totalbatches = len(trainloader)

# Initialize the model, loss function, and optimizer

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



## THIS IS TO VISUALIZE THE TRAINING DATA
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# images = images[:10]
# labels = labels[:10]

# fig, axes = plt.subplots(1, 10, figsize=(18, 3))
# for i in range(10): 
#     axes[i].imshow(images[i].squeeze(0), cmap="gray")
#     axes[i].set_title(f"Label = {labels[i].item()}")
#     axes[i].axis("off")
# plt.show()





print("Training")
epochs = 1
for epoch in range(epochs): 
    batch = 0 
    for images, labels in trainloader:
        batch += 1
        if batch % math.floor((totalbatches / 4)) == 0 :
            print(f"Epoch {epoch} batch progress: {round(batch*100/totalbatches)}% ")
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training complete.")







def evaluate(model, dataloader, device):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation needed
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total number of samples

    accuracy = correct / total * 100
    return accuracy

# Run evaluation
accuracy = evaluate(model, testloader, device)
print(f"Test Accuracy: {accuracy:.2f}%")






dataiter = iter(testloader)
images, labels = next(dataiter)
images = images[:10]
labels = labels[:10]

model.eval()
with torch.no_grad():
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    _, preds = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)

images, labels, preds, probabilities = images.cpu(), labels.cpu(), preds.cpu(), probabilities.cpu()

fig, axes = plt.subplots(1, 10, figsize=(18,3))
for i in range(10):
    axes[i].imshow(images[i].squeeze(0), cmap="gray")
    axes[i].set_title(f"True: {labels[i].item()}\nPred: {preds[i].item()}\nProb: {probabilities[i][preds[i]].item():.2f}")
    axes[i].axis("off")
plt.show()
