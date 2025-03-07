from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from sandbox.toolbox.loss_functions import TD_KD
from sandbox.toolbox.data_loader import Cifar10, Cifar100

from sandbox.toolbox.utils import get_path, plot_the_things, evaluate_model

import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

device = 'cuda'

################## SPECIFY SETTINGS ######################
Epochs = 150 
Data = Cifar100
Student = ResNetBaby 
Teacher = ResNet112

expirement_small_name = None

################## INITIALIZING THE THINGS ######################
Data = Data()
Distillation = TD_KD()

trainloader, testloader = Data.trainloader, Data.testloader

Student = Student(Data.class_num).to(device)
Teacher = Teacher(Data.class_num).to(device)
checkpoint = torch.load(f'models/{Data.name}_{Teacher.model_type}.pth', weights_only=True)
Teacher.load_state_dict(checkpoint['weights'])

experiment_name = f'{Data.name}/{Student.model_type}_{Teacher.model_type}/{Distillation.name}'
expirement_small_name, path = get_path(experiment_name, expirement_small_name)
model_name = f'{Data.name}_{Student.model_type}'

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)
criterion = nn.L1Loss()

train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

BETA = 125

tl.set_backend('pytorch')
def tucker_decomposition(feature_maps, rank):
    batch_size, num_channels, height, width = feature_maps.shape
    ranks = [batch_size, rank, height, width]

    core, factors = tl.decomposition.tucker(
        feature_maps, rank=ranks)

    x_reconstructed = tl.tucker_to_tensor((core, factors))
    x_reconstructed = x_reconstructed.to(feature_maps.device)

    return x_reconstructed

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

for epoch in range(Epochs):
    print(f'{epoch=}')
    Teacher.eval()
    Student.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        ###################################################################################
        teacher_outputs = Teacher(inputs)
        student_outputs = Student(inputs)

        teacher_features3 = tucker_decomposition(teacher_outputs[2], 32)
        student_features3 = tucker_decomposition(student_outputs[2], 32)

        loss = BETA * (criterion(FT(student_features3), FT(teacher_features3.detach()))) + F.cross_entropy(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    scheduler.step()

    print(f'TRAIN | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(correct*100/total)


    tel, ta = evaluate_model(Student, testloader)
    test_loss.append(tel)
    test_acc.append(ta)

    if ta > max_acc:
        max_acc = ta
        torch.save({'weights': Student.state_dict()}, f'{path}/{model_name}.pth')

    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_name, expirement_small_name, path)


logs = {
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_loss': test_loss,
    'test_acc': test_acc
}

with open(f'{path}/logs.pkl', 'wb') as f:
    pickle.dump(logs, f)








