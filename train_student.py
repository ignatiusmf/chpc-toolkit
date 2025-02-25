import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from models import ResNet112, ResNet56 
import torch.nn.functional as F

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
checkpoint = torch.load("checkpoint/Teacher.pth", weights_only=True)
Teacher.load_state_dict(checkpoint['model_state_dict'])

Student_control = ResNet56().to(device)
optimizer_control = optim.SGD(Student_control.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler_control = optim.lr_scheduler.CosineAnnealingLR(optimizer_control, T_max=100)






criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



print(f"Total parameters for large model: {sum(p.numel() for p in Teacher.parameters()):,}")
print(f"Total parameters for small model: {sum(p.numel() for p in Student_control.parameters()):,}")

max_acc = 0.0 ## TODO 
loss = [[],[],[],[],[]]
accuracy = [[],[],[],[],[]]

def vanilla(outputs, outputs_teacher, targets):
    loss = criterion(outputs[3], targets)
    return loss

def logits_kd(outputs, outputs_teacher, targets, T=4.0, alpha=0.7):
    soft_targets = F.kl_div(
        F.log_softmax(outputs[3] / T, dim=1),
        F.softmax(outputs_teacher[3] / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_targets = criterion(outputs[3], targets)
    return alpha * soft_targets + (1 - alpha) * hard_targets

def features_kd():
    print("yeet")

def td_kd():
    print("yeet")

def image_denoise_kd():
    print("yeet")

def train(Teacher, Student, student_loss, optimizer, scheduler):
    Teacher.eval()
    Student.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = Student(inputs)
        with torch.no_grad():
            outputs_teacher = Teacher(inputs)

        loss = student_loss(outputs, outputs_teacher, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total

    print(f"Loss: {avg_loss:.3f} | Train accuracy: {accuracy:.3f}% |")

    scheduler.step()

    return avg_loss, accuracy

def test(Student):
    Student.eval()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = Student(inputs)

        loss = F.cross_entropy(outputs[3], targets)
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
    tei = np.array(testi)

    plt.plot(np.log10(tei[1:,0]), label="Test loss") # np.log10(tei[:,0])

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/Loss.png")
    plt.close()

    plt.plot(tei[1:,1], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("logs/Accuracy.png")
    plt.close()

print("Teacher test", test(Teacher))
print("Training started")
for epoch in range(3):
    train_loss, train_accuracy = train(Teacher, Student_control, vanilla_loss, optimizer_control, scheduler_control)
    test_loss, test_accuracy = test(Student_control)
    plot()

    if test_accuracy > max_acc: ## TODO
        max_acc = test_accuracy
        checkpoint = {
            'model_state_dict': Student_control.state_dict(),
        }
        torch.save(checkpoint, f"checkpoint/Student_{epoch}_{test_accuracy:.0f}.pth")




