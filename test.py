import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(16*28*28, 10)
        )

    def forward(self, x):
        return self.network(x) 

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



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




# Training loop
epochs = 1
for epoch in range(epochs):  # One epoch for simplicity
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training complete.")

dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images[:10]
labels = labels[:10]





# Load the test dataset
testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Evaluation function
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
