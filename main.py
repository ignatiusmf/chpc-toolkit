import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

def visualize_dataset(dataset, dataloader):
    class_names = dataset.classes
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images[:10]
    labels = labels[:10]

    fig, axes = plt.subplots(1, 10, figsize=(18, 3))
    for i in range(10):
        img = images[i].permute(1, 2, 0) * 0.5 + 0.5  
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[labels[i].item()]}")
        axes[i].axis("off")
    plt.show()

def visualize_predictions(model, dataset, dataloader, device):
    class_names = dataset.classes
    dataiter = iter(dataloader)
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

    fig, axes = plt.subplots(1, 10, figsize=(18, 3))
    for i in range(10):
        img = images[i].permute(1, 2, 0) * 0.5 + 0.5  
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[labels[i].item()]}\nPred: {class_names[preds[i].item()]}\nProb: {probabilities[i][preds[i]].item():.2f}")
        axes[i].axis("off")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=data_transforms, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=data_transforms, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    totalbatches = len(trainloader)


    # visualize_dataset(trainset, trainloader)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add L2 regularization

    print("Training")
    epochs = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        for images, labels in trainloader:
            batch += 1
            # if batch % math.floor((totalbatches / 3)) == 0:
            #     print(f"Epoch {epoch+1} batch progress: {round(batch*100/totalbatches)}% ")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")

    # Evaluation function
    def evaluate(model, dataloader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        return accuracy

    # Run evaluation
    accuracy = evaluate(model, testloader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # visualize_predictions(model, testset, testloader, device)


main()