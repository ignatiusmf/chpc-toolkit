import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import models

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

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion_CE = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"Total parameters for large model: {sum(p.numel() for p in Teacher.parameters()):,}")

max_acc = 0.0

def train(model, epoch):
    print(f'{epoch=}')
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion_CE(outputs[3], targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate for previous epoch: {current_lr:.10f}")

    avg_loss = running_loss / len(trainloader)
    trainset_acc = 100 * correct / total

    print(f"Loss: {.3f} | Train accuracy: {trainset_acc:.3f}% |")

    scheduler.step()

    return avg_loss, correct / total




for epoch in range(100):
    train(model, epoch)
