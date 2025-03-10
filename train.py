from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from sandbox.toolbox.data_loader import Cifar10, Cifar100
from sandbox.toolbox.utils import get_names, plot_the_things, evaluate_model, get_settings

import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle

device = 'cuda'

################## SPECIFY SETTINGS ######################
settings = get_settings()

Epochs = settings['Epochs']
Data = settings['Data']
Student = settings['Student']
experiment_id = settings['experiment_id']

################## INITIALIZING THE THINGS ######################
Data = Data()

trainloader, testloader = Data.trainloader, Data.testloader
Student = Student(Data.class_num).to(device)

experiment_name, experiment_id, model_name, path = get_names(Data.name, Student.model_type, experiment_id=experiment_id) 

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0

for epoch in range(Epochs):
    print(f'{epoch=}')
    Student.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = Student(inputs)
        loss = F.cross_entropy(outputs[3], targets, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    scheduler.step()

    print(f'TRAIN | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.2f} |')
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(100*correct/total)

    tl, ta = evaluate_model(Student, testloader)
    test_loss.append(tl)
    test_acc.append(ta)

    if ta > max_acc:
        max_acc = ta
        torch.save({'weights': Student.state_dict()}, f'{path}/{model_name}.pth')

    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_name, experiment_id, path)

logs = {
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_loss': test_loss,
    'test_acc': test_acc
}

with open(f'{path}/logs.pkl', 'wb') as f:
    pickle.dump(logs, f)