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
        if optimizer:
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

    model_teacher = LargeCNN().to(device)
    model_small = SmallCNN().to(device)
    model_student = SmallCNN().to(device)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_small = optim.Adam(model_small.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_student = optim.Adam(model_student.parameters(), lr=0.001, weight_decay=1e-4)

    total_params_teacher = sum(p.numel() for p in model_teacher.parameters())
    print(f"Total parameters for teacher model: {total_params_teacher:,}")
    total_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Total parameters for small / student model: {total_params_small:,}")

    load_checkpoint(model_teacher, None, "checkpoint/large_cnn_checkpoint.pth")
    start_epoch_small = load_checkpoint(model_small, optimizer_small, "checkpoint/kd_small.pth")
    start_epoch_student = load_checkpoint(model_student, optimizer_student, "checkpoint/kd_student.pth")

    lossi = load_loss("checkpoint/kd.json")
    test_accuracy = load_acc("checkpoint/kd_test_acc.json")

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

    epochs = 5 
    for epoch in range(start_epoch_small, epochs):
        model_small.train()
        model_student.train()
        running_loss_small = 0.0
        running_loss_student = 0.0
        batch = 0
        for images, labels in trainloader:
            batch += 1
            if batch % math.floor((len(trainloader) / 10)) == 0 and epochs == 1:
                print(f"Epoch {epoch+1} batch progress: {round(batch*100/len(trainloader))}% ")

            images, labels = images.to(device), labels.to(device)
            
            optimizer_small.zero_grad()
            outputs_small = model_small(images)
            loss_small = criterion(outputs_small, labels)
            loss_small.backward()
            optimizer_small.step()
            running_loss_small += loss_small.item()

            optimizer_student.zero_grad()
            outputs_student= model_student(images)
            loss_student= criterion(outputs_student, labels)
            loss_student.backward()
            optimizer_student.step()
            running_loss_student+= loss_student.item()

            lossi.append([np.log10(loss_small.item()), np.log10(loss_student.item())])
        
        avg_loss_small = running_loss_small / len(trainloader)
        avg_loss_student= running_loss_student / len(trainloader)

        print(f"Epoch [{epoch+1}/{epochs}], SmallCNN Loss: {avg_loss_small:.4f}, StudentCNN Loss: {avg_loss_student:.4f}")

        accuracy_small= evaluate(model_small, testloader, device)
        accuracy_student = evaluate(model_student, testloader, device)

        print(f"Test Accuracy - SmallCNN: {accuracy_small:.2f}%")
        print(f"Test Accuracy - StudentCNN: {accuracy_student:.2f}%")

        test_accuracy.append([accuracy_small, accuracy_student])

        save_checkpoint(model_small, optimizer_small, epoch + 1, "checkpoint/kd_small.pth")
        save_checkpoint(model_student, optimizer_student, epoch + 1, "checkpoint/kd_student.pth")

        save_loss(lossi, "checkpoint/kd_loss.json")
        save_acc(test_accuracy, "checkpoint/kd_test_acc.json")


    print("Training complete.")

    lossi = np.array(lossi)

    plt.plot(np.convolve(lossi[:,0], np.ones(100)/100, mode='valid'), label="SmallCNN")
    plt.plot(np.convolve(lossi[:,1], np.ones(100)/100, mode='valid'), label="StudentCNN")
    for x in range(0, len(lossi), len(trainloader)):
        plt.axvline(x=x, color='gray', linestyle='--',linewidth=0.5)

    plt.xlabel("Batch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/KD_Loss.png")
    plt.close()


    test_accuracy = np.array(test_accuracy)
    plt.plot(test_accuracy[:,0], label="SmallCNN Accuracy")
    plt.plot(test_accuracy[:,1], label="StudentCNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("KD Test Accuracy")
    plt.legend()
    plt.savefig("logs/KD_test_acc.png")
    plt.close()


main()