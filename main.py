import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import json

# Define a larger CNN model
class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.network(x)

# Define a smaller CNN model
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256, 10)
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
    plt.savefig("logs/dataset_visualization.png")
    plt.close()

def visualize_predictions(model, dataset, dataloader, device, description):
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
    plt.savefig(f"logs/model_predictions-{description}.png")
    plt.close()

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {filename} (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print("No checkpoint found, starting from scratch.")
        return 0


def save_loss(loss_data, filename):
    with open(filename, 'w') as f:
        json.dump(loss_data, f)

def load_loss(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            loss_data = json.load(f)
        print(f"Loss data loaded from {filename}")
        return loss_data
    else:
        return []

def save_acc(acc_data, filename):
    with open(filename, 'w') as f:
        json.dump(acc_data, f)

def load_acc(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            acc_data = json.load(f)
        print(f"Test accuracy data loaded from {filename}")
        return acc_data
    else:
        return []


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

    visualize_dataset(trainset, trainloader)

    model_large = LargeCNN().to(device)
    model_small = SmallCNN().to(device)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_large = optim.Adam(model_large.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_small = optim.Adam(model_small.parameters(), lr=0.001, weight_decay=1e-4)

    total_params_large = sum(p.numel() for p in model_large.parameters())
    print(f"Total parameters for large model: {total_params_large:,}")
    total_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Total parameters for small model: {total_params_small:,}")

    start_epoch_large = load_checkpoint(model_large, optimizer_large, "checkpoint/large_cnn_checkpoint.pth")
    start_epoch_small = load_checkpoint(model_small, optimizer_small, "checkpoint/small_cnn_checkpoint.pth")

    lossi = load_loss("checkpoint/loss.json")

    test_accuracy = load_acc("checkpoint/test_acc.json")

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

    epochs = 50
    for epoch in range(max(start_epoch_large, start_epoch_small), epochs):
        model_large.train()
        model_small.train()
        running_loss_large = 0.0
        running_loss_small = 0.0
        batch = 0
        for images, labels in trainloader:
            batch += 1
            if batch % math.floor((len(trainloader) / 10)) == 0 and epochs == 1:
                print(f"Epoch {epoch+1} batch progress: {round(batch*100/len(trainloader))}% ")

            images, labels = images.to(device), labels.to(device)
            
            optimizer_large.zero_grad()
            outputs_large = model_large(images)
            loss_large = criterion(outputs_large, labels)
            loss_large.backward()
            optimizer_large.step()
            running_loss_large += loss_large.item()

            optimizer_small.zero_grad()
            outputs_small = model_small(images)
            loss_small = criterion(outputs_small, labels)
            loss_small.backward()
            optimizer_small.step()
            running_loss_small += loss_small.item()

            lossi.append([np.log10(loss_large.item()), np.log10(loss_small.item())])
        
        avg_loss_large = running_loss_large / len(trainloader)
        avg_loss_small = running_loss_small / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}], LargeCNN Loss: {avg_loss_large:.4f}, SmallCNN Loss: {avg_loss_small:.4f}")

        accuracy_large = evaluate(model_large, testloader, device)
        accuracy_small = evaluate(model_small, testloader, device)
        print(f"Test Accuracy - LargeCNN: {accuracy_large:.2f}%")
        print(f"Test Accuracy - SmallCNN: {accuracy_small:.2f}%")
        test_accuracy.append([accuracy_large, accuracy_small])

        save_checkpoint(model_large, optimizer_large, epoch + 1, "checkpoint/large_cnn_checkpoint.pth")
        save_checkpoint(model_small, optimizer_small, epoch + 1, "checkpoint/small_cnn_checkpoint.pth")
        save_loss(lossi, "checkpoint/loss.json")
        save_acc(test_accuracy, "checkpoint/test_acc.json")


    print("Training complete.")

    lossi = np.array(lossi)

    plt.plot(np.convolve(lossi[:,0], np.ones(100)/100, mode='valid'), label="LargeCNN")
    plt.plot(np.convolve(lossi[:,1], np.ones(100)/100, mode='valid'), label="SmallCNN")
    for x in range(0, len(lossi[:,0]), len(trainloader)):
        plt.axvline(x=x, color='gray', linestyle='--',linewidth=0.5)

    plt.xlabel("Batch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/Loss.png")
    plt.close()


    test_accuracy = np.array(test_accuracy)
    plt.plot(test_accuracy[:,0], label="LargeCNN Accuracy")
    plt.plot(test_accuracy[:,1], label="SmallCNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("logs/test_acc.png")
    plt.close()


    visualize_predictions(model_large, testset, testloader, device, "Large Model")
    visualize_predictions(model_small, testset, testloader, device, "Small Model")


main()