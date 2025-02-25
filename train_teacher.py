import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from models import ResNet112

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)


Teacher = ResNet112().to(device)

optimizer = optim.SGD(Teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_test = nn.CrossEntropyLoss()
print(f"Total parameters for large model: {sum(p.numel() for p in Teacher.parameters()):,}")

max_acc = 0.0
traini = [[0,0]]
testi = [[0,0]]

def train():
    Teacher.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = Teacher(inputs)
        loss = criterion(outputs[3], targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate for previous epoch: {current_lr:.10f}")

    avg_loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total

    print(f"Loss: {avg_loss:.3f} | Train accuracy: {accuracy:.3f}% |")

    scheduler.step()

    return avg_loss, accuracy

def test():
    Teacher.eval()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = Teacher(inputs)

        loss = criterion_test(outputs[3], targets)
        running_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)

        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

    avg_loss = running_loss / len(testloader)
    accuracy = 100 * correct / total
    print(f"Loss: {avg_loss:.3f} | Test accuracy: {accuracy:.3f}% |")

    return avg_loss, accuracy

import matplotlib.pyplot as plt
import numpy as np
def plot():
    tri = np.array(traini)
    tei = np.array(testi)

    plt.plot(tri[:,0], label="Train_loss") # np.log10(tri[:,0])
    plt.plot(tei[:,0], label="Test_loss") # np.log10(tei[:,0])

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/Loss.png")
    plt.close()

    plt.plot(tri[:,1], label="Train Accuracy")
    plt.plot(tei[:,1], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("logs/Accuracy.png")
    plt.close()


print("Trianing started")
for epoch in range(3):
    train_loss, train_accuracy = train()
    traini.append([train_loss, train_accuracy])
    test_loss, test_accuracy = test()
    testi.append([test_loss, test_accuracy])
    plot()

    if test_accuracy > max_acc:
        max_acc = test_accuracy
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': Teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, f"checkpoint/Teacher_{epoch}_{test_accuracy:.0f}.pth")




