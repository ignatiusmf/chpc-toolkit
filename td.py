from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.loss_functions import Vanilla, Logits_KD, Factor_Transfer_KD, TD_KD
from toolbox.data_loader import Cifar10, Cifar100

from toolbox.utils import get_path, plot_the_things

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

################## SPECIFY SETTINGS ######################
Epochs = 150
Data = Cifar100
Student = ResNet56 
Teacher = ResNet112
Distillation = TD_KD 

expirement_small_name = None

################## INITIALIZING THE THINGS ######################
Data = Data()
Distillation = Distillation()

trainloader, testloader = Data.trainloader, Data.testloader

Student = Student(Data.class_num).to(device)
Teacher = Teacher(Data.class_num).to(device)

checkpoint = torch.load(f'models/{Data.name}_{Teacher.model_type}.pth', weights_only=True)
Teacher.load_state_dict(checkpoint['weights'])

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)

criterion_CE = nn.CrossEntropyLoss()
criterion = nn.L1Loss()

train_loss = []
train_acc = []
test_loss = []
test_acc = []

max_acc = 0.0

experiment_name = f'{Data.name}/{Student.model_type}_{Teacher.model_type}/{Distillation.name}'
expirement_small_name, path = get_path(experiment_name, expirement_small_name)

BETA = 125

import tensorly as tl
tl.set_backend('pytorch')
def tucker_decomposition(feature_maps, rank):
    batch_size, num_channels, height, width = feature_maps.shape
    ranks = [batch_size, rank, height, width]

    # Decompose the tensor
    core, factors = tl.decomposition.tucker(
        feature_maps, rank=ranks)

    x_reconstructed = tl.tucker_to_tensor((core, factors))
    x_reconstructed = x_reconstructed.to(feature_maps.device)

    return x_reconstructed


def eval():
    Student.eval()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs= Student(inputs)
        loss = criterion_CE(outputs[3], targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    print('Loss: %.3f | Acc net: %.3f%%' % (val_loss / (b_idx + 1), 100. * correct / total))
    test_loss.append(val_loss/(b_idx+1))
    test_acc.append(correct*100/total)

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

        # Knowledge transfer with FT loss at the last layer
        ###################################################################################
        teacher_outputs = Teacher(inputs)
        student_outputs = Student(inputs)

        teacher_features3 = tucker_decomposition(teacher_outputs[2], 32)
        student_features3 = tucker_decomposition(student_outputs[2], 32)

        loss = BETA * (criterion(FT(student_features3), FT(teacher_features3.detach()))) + criterion_CE(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()

        val_loss += loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Loss: %.3f | Acc net: %.3f%%|' % (val_loss / (b_idx + 1), 100. * correct / total))
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(correct*100/total)

    scheduler.step()

    eval()

    plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_name, expirement_small_name, path)










