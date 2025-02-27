import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from models import ResNet112, ResNet56, ResNetBaby, ResNet20
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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)


Teacher = ResNetBaby(100).to(device)
# checkpoint = torch.load("checkpoint/Teacher.pth", weights_only=True)
# Teacher.load_state_dict(checkpoint['model_state_dict'])

Student_vanilla = ResNetBaby(100).to(device)
optimizer_vanilla = optim.SGD(Student_vanilla.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler_vanilla = optim.lr_scheduler.CosineAnnealingLR(optimizer_vanilla, T_max=100)

Student_logits_kd = ResNetBaby(100).to(device)
optimizer_logits_kd = optim.SGD(Student_logits_kd.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler_logits_kd = optim.lr_scheduler.CosineAnnealingLR(optimizer_logits_kd, T_max=100)




criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



print(f"Total parameters for large model: {sum(p.numel() for p in Teacher.parameters()):,}")
print(f"Total parameters for small model: {sum(p.numel() for p in Student_vanilla.parameters()):,}")

max_acc = 0.0 ## TODO 


def vanilla(outputs, outputs_teacher, targets):
    loss = criterion(outputs[3], targets)
    return loss

def logits_kd(outputs, outputs_teacher, targets, T=4.0, alpha=0.7):
    soft_targets = F.kl_div(
        F.log_softmax(outputs[3] / T, dim=1),
        F.softmax(outputs_teacher[3] / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_targets = F.cross_entropy(outputs[3], targets)
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

    print(f"Learning rate this epoch: {optimizer.param_groups[-1]['lr']:.10f}")

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

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]  
    for i, (key, values) in enumerate(model_logs.items()):
        color = color_cycle[i % len(color_cycle)]  
        for loss_name in ["test_loss", "train_loss"]:
            linestyle = 'dotted' if loss_name == "train_loss" else 'solid'
            plt.plot(np.log10(np.array(values[loss_name])), linestyle=linestyle, color=color, label=f'{key} {loss_name}')

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("logs/Loss.png")
    plt.close()


    for i, (key, values) in enumerate(model_logs.items()):
        color = color_cycle[i % len(color_cycle)] 
        for acc_name in ["test_acc", "train_acc"]:
            linestyle = 'dotted' if acc_name == "train_acc" else 'solid'
            plt.plot(np.array(values[acc_name]), linestyle=linestyle, color=color, label=f'{key} {acc_name}')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("logs/Accuracy.png")
    plt.close()








print("Teacher test loss / acc:")
test(Teacher)
print("Training started")

models = {
    "vanilla": [Teacher, Student_vanilla, vanilla, optimizer_vanilla, scheduler_vanilla],
    "logits_kd": [Teacher, Student_logits_kd, logits_kd, optimizer_logits_kd, scheduler_logits_kd]
}

model_logs = {
    "vanilla": {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    },
    "logits_kd": {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
}
class Model:
    def __init__(self, teacher, student, loss_fn, optimizer, scheduler):
        self.teacher = teacher
        self.student = student
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def update_logs(self, train_loss, train_acc, test_loss, test_acc):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)

def model_trainer(model_name):
    print(f'{model_name=}')
    trl, tra = train(*models[model_name])
    tel, tea = test(models[model_name][1])

    model_logs[model_name]["train_loss"].append(trl)
    model_logs[model_name]["train_acc"].append(tra)
    model_logs[model_name]["test_loss"].append(tel)
    model_logs[model_name]["test_acc"].append(tea)



for epoch in range(10):
    print(f'{epoch=}')
    model_trainer("vanilla")
    model_trainer("logits_kd")
    plot()


    # if test_accuracy > max_acc: ## TODO
    #     max_acc = test_accuracy
    #     checkpoint = {
    #         'model_state_dict': Student_vanilla.state_dict(),
    #     }
    #     torch.save(checkpoint, f"checkpoint/Student_{epoch}_{test_accuracy:.0f}.pth")




