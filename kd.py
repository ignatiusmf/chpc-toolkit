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

class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  

            nn.AdaptiveAvgPool2d(1),  # Global pooling for better generalization
            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),

            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),

            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)


# Define a smaller CNN model
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d(1),  # Replaces Flatten() + large FC layers
            nn.Flatten(),

            nn.Linear(256, 512),  # Increased FC layer
            nn.LeakyReLU(0.1),

            nn.Dropout(0.4),
            nn.Linear(512, 10)
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
    return correct / total * 100

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft_targets = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_targets = F.cross_entropy(student_logits, labels)
    return alpha * soft_targets + (1 - alpha) * hard_targets

def train_student_with_kd_and_control(model_student, model_control, model_teacher, trainloader, optimizer_student, optimizer_control, device, T=4.0, alpha=0.7):
    model_student.train()
    model_control.train()
    model_teacher.eval()
    running_loss_student = 0.0
    running_loss_control = 0.0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer_student.zero_grad()
        optimizer_control.zero_grad()
        
        with torch.no_grad():
            outputs_teacher = model_teacher(images)
        
        outputs_student = model_student(images)
        outputs_control = model_control(images)
        
        loss_student = distillation_loss(outputs_student, outputs_teacher, labels, T, alpha)
        loss_control = F.cross_entropy(outputs_control, labels)
        
        loss_student.backward()
        optimizer_student.step()
        
        loss_control.backward()
        optimizer_control.step()
        
        running_loss_student += loss_student.item()
        running_loss_control += loss_control.item()
    
    return running_loss_student / len(trainloader), running_loss_control / len(trainloader)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model_teacher = LargeCNN().to(device)
    model_student = SmallCNN().to(device)
    model_control = SmallCNN().to(device)

    optimizer_student = optim.Adam(model_student.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_control = optim.Adam(model_control.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_student = optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, mode='min', factor=0.1, patience=2)
    scheduler_control = optim.lr_scheduler.ReduceLROnPlateau(optimizer_control, mode='min', factor=0.1, patience=2)

    total_params_teacher = sum(p.numel() for p in model_teacher.parameters())
    print(f"Total parameters for teacher model: {total_params_teacher:,}")
    total_params_small = sum(p.numel() for p in model_student.parameters())
    print(f"Total parameters for control / student model: {total_params_small:,}")

    load_checkpoint(model_teacher, None, "checkpoint/large_cnn_checkpoint.pth")
    start_epoch_large = load_checkpoint(model_student, optimizer_student, "checkpoint/KD_student.pth")
    start_epoch_small = load_checkpoint(model_control, optimizer_control, "checkpoint/KD_control.pth")

    lossi = load_loss("checkpoint/KD_loss.json")

    test_accuracy = load_acc("checkpoint/KD_test_acc.json")

    T = 4.0
    alpha = 0.7

    epochs = 100
    for epoch in range(epochs):

        model_student.train()
        model_control.train()
        model_teacher.eval()
        running_loss_student = 0.0
        running_loss_control = 0.0
        
        batch = 0
        for images, labels in trainloader:
            batch += 1
            if batch % math.floor((len(trainloader) / 10)) == 0 and epochs == 1:
                print(f"Epoch {epoch+1} batch progress: {round(batch*100/len(trainloader))}% ")

            images, labels = images.to(device), labels.to(device)
            
            optimizer_student.zero_grad()
            optimizer_control.zero_grad()
            
            with torch.no_grad():
                outputs_teacher = model_teacher(images)
            
            outputs_student = model_student(images)
            outputs_control = model_control(images)
            
            loss_student = distillation_loss(outputs_student, outputs_teacher, labels, T, alpha)
            loss_control = F.cross_entropy(outputs_control, labels)
            
            loss_student.backward()
            optimizer_student.step()
            
            loss_control.backward()
            optimizer_control.step()
            
            running_loss_student += loss_student.item()
            running_loss_control += loss_control.item()

            lossi.append([np.log10(loss_student.item()), np.log10(loss_control.item())])

        loss_student, loss_control = running_loss_student / len(trainloader), running_loss_control / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}], Student Loss: {loss_student:.4f}, Control Loss: {loss_control:.4f}")


        current_lr_student = optimizer_student.param_groups[0]['lr']
        current_lr_control = optimizer_control.param_groups[0]['lr']
        print(f"Before LR Step -> StudentCNN LR: {current_lr_student:.10f}, ControlCNN LR: {current_lr_control:.10f}")

        # Update learning rate based on validation loss
        scheduler_student.step(loss_student)
        scheduler_control.step(loss_control)



        acc_student = evaluate(model_student, testloader, device)
        acc_control = evaluate(model_control, testloader, device)
        print(f"Test Accuracy - StudentCNN: {acc_student:.2f}%")
        print(f"Test Accuracy - ControlCNN: {acc_control:.2f}%")
        test_accuracy.append([acc_student, acc_control])

        plot_accuracy(test_accuracy)
        plot_loss(lossi, trainloader)

        save_checkpoint(model_student, optimizer_student, epoch + 1, "checkpoint/KD_student.pth")
        save_checkpoint(model_control, optimizer_control, epoch + 1, "checkpoint/KD_control.pth")
        save_loss(lossi, "checkpoint/KD_loss.json")
        save_acc(test_accuracy, "checkpoint/KD_test_acc.json")


    print("Training complete.")

def plot_loss(loss,loader):
    lossi = np.array(loss)

    plt.plot(np.convolve(lossi[:,0], np.ones(100)/100, mode='valid'), label="Student")
    plt.plot(np.convolve(lossi[:,1], np.ones(100)/100, mode='valid'), label="Control")
    for x in range(0, len(lossi[:,0]), len(loader)):
        plt.axvline(x=x, color='gray', linestyle='--',linewidth=0.5)

    plt.xlabel("Batch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/KD_Loss.png")
    plt.close()

def plot_accuracy(accuracy):
    test_accuracy = np.array(accuracy)
    plt.plot(test_accuracy[:,0], label="StudentCNN Accuracy")
    plt.plot(test_accuracy[:,1], label="ControlCNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("logs/KD_test_acc.png")
    plt.close()

main()
