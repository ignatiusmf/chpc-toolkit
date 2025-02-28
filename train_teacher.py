import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from toolbox.models import ResNet112, ResNet20
from toolbox.data_loader import get_loaders

device = 'cuda'

trainloader, testloader = get_loaders('cifar100', 0)

Teacher = ResNet20(100).to(device)

optimizer = optim.SGD(Teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
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

    print(f"Learning rate for previous epoch: {optimizer.param_groups[-1]['lr']:.10f}")

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


print("Training started")
for epoch in range(150):
    print(f'{epoch=}')
    train_loss, train_accuracy = train()
    traini.append([train_loss, train_accuracy])
    test_loss, test_accuracy = test()
    testi.append([test_loss, test_accuracy])
    plot()


    if test_accuracy > max_acc:
        max_acc = test_accuracy
        checkpoint = {
            'model_state_dict': Teacher.state_dict()
        }
        torch.save(checkpoint, f"checkpoint/RN112_C100_E150_{epoch}_{test_accuracy:.0f}_CA.pth")




