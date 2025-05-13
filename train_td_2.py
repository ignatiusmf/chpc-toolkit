from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from sandbox.toolbox.data_loader import Cifar10, Cifar100

from sandbox.toolbox.utils import get_names, plot_the_things, evaluate_model, get_settings

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
Student = ResNet56
Teacher = ResNet112
expirement_id = "Full_FM_TD"

################## INITIALIZING THE THINGS ######################
Data = Data()

trainloader, testloader = Data.trainloader, Data.testloader

Student = Student(Data.class_num).to(device)
Teacher = Teacher(Data.class_num).to(device)
checkpoint = torch.load(f'models/{Data.name}_{Teacher.model_type}.pth', weights_only=True)
Teacher.load_state_dict(checkpoint['weights'])

experiment_name, experiment_id, model_name, path = get_names(Data.name, Student.model_type, Teacher.model_type, 'td', expirement_id) 

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)
criterion = nn.L1Loss()

train_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

BETA = 125

tl.set_backend("pytorch")
def tucker(feature_map):
   batch_size, channels, height, width = feature_map.shape
   core, factors = tl.decomposition.tucker(feature_map, rank=[batch_size, 32, 8, 8])
   return core

# def tucker(feature_map):
#     batch_size, channels, height, width = feature_map.shape  # [128, 64, 8, 8]
#     core, factors = tl.decomposition.tucker(
#         feature_map,
#         rank=[32, 16, 4, 4],
#         svd='randomized_svd',  # Robust SVD method
#         init='random',        # Avoid SVD-based init
#         tol=1e-5,             # Reasonable convergence tolerance
#         n_iter_max=200        # Allow more iterations if needed
#     )
#     return core

def tucker_sample_split(feature_map):
    core_tensor_list = []
    for i in range(128):
        single_fmap = feature_map[i].unsqueeze(0)
        core = tucker(single_fmap)[0]
        core_tensor_list.append(core)
        
    combined_tensor = torch.stack(core_tensor_list, dim=0)
    return combined_tensor

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

for epoch in range(Epochs):
    print(f'{epoch=}')
    Teacher.eval()
    Student.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 10 == 0:
            print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        ###################################################################################
        teacher_outputs = Teacher(inputs)
        student_outputs = Student(inputs)

        teacher_features3 = tucker(teacher_outputs[2])
        student_features3 = tucker(student_outputs[2])

        loss = BETA * (criterion(FT(student_features3), FT(teacher_features3.detach()))) + F.cross_entropy(student_outputs[3], targets)
        print(loss)
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

    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_name, expirement_id, path)


logs = {
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_loss': test_loss,
    'test_acc': test_acc
}

with open(f'{path}/logs.pkl', 'wb') as f:
    pickle.dump(logs, f)


