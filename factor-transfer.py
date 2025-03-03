from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.loss_functions import Vanilla, Logits_KD, Factor_Transfer_KD
from toolbox.data_loader import Cifar10, Cifar100

from toolbox.factor_transfer_components import Paraphraser, Translator
from toolbox.utils import get_path

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
Distillation = Factor_Transfer_KD 

expirement_small_name = 'test' 

################## INITIALIZING THE THINGS ######################
Data = Data()
Distillation = Distillation()

trainloader, testloader = Data.trainloader, Data.testloader

Student = Student(Data.class_num).to(device)
Teacher = Teacher(Data.class_num).to(device)

checkpoint = torch.load(f'models/{Data.name}_{Teacher.model_type}.pth', weights_only=True)
Teacher.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)

criterion_CE = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
test_loss = []
test_acc = []

max_acc = 0.0

experiment_name = f'{Data.name}/{Student.model_type}_{Teacher.model_type}/{Distillation.name}'
expirement_small_name, path = get_path(experiment_name, expirement_small_name)

################## FACTOR TRANSFER INITIALIZATION ##################

Paraphraser_Epochs = 1
paraphraser_compression = 0.5
train_paraphraser = True

Paraphraser = Paraphraser(64, int(round(64*paraphraser_compression))).to(device)
paraphraser_optimizer = optim.SGD(Paraphraser.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
paraphraser_scheduler = optim.lr_scheduler.CosineAnnealingLR(paraphraser_optimizer, T_max=Paraphraser_Epochs)
criterion = nn.L1Loss()

if not train_paraphraser:
    checkpoint = torch.load(f'{path}/paraphraser.pth')
else:
    for epoch in range(Paraphraser_Epochs):
        print(f'{epoch=}')
        Teacher.eval()
        Paraphraser.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            paraphraser_optimizer.zero_grad()
            outputs= Teacher(inputs)
            output_p = Paraphraser(outputs[2],0)
            loss = criterion(output_p, outputs[2].detach())
            loss.backward()
            paraphraser_optimizer.step()
            train_loss += loss.item()
            b_idx = batch_idx
        print('Loss: %.3f | ' % (train_loss / (b_idx + 1)))
        paraphraser_scheduler.step()
        checkpoint = {'weights': Paraphraser.state_dict()}
        torch.save(checkpoint, f'path/paraphraser.pth')

################## FACTOR TRANSFER TRAINING ##################

BETA = 500
Translator = Translator(64, int(round(64*paraphraser_compression))).to(device)
translator_optimizer = optim.SGD(Translator.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
translator_scheduler = optim.lr_scheduler.CosineAnnealingLR(translator_optimizer, T_max=Epochs)

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
    print('Loss: %.3f | Acc net: %.3f%%' % (train_loss / (b_idx + 1), 100. * correct / total))
    return val_loss / (b_idx + 1),  correct / total

def FT(x):
    return F.normalize(x.view(x.size(0), -1))

for epoch in range(Epochs):
    Teacher.eval()
    Paraphraser.eval()
    Student.train()
    Translator.train()
    train_loss, correct, total = 0, 0, 0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        translator_optimizer.zero_grad()

        # Knowledge transfer with FT loss at the last layer
        ###################################################################################
        teacher_outputs = Teacher(inputs)
        student_outputs = Student(inputs)

        factor_t = Paraphraser(teacher_outputs[2],1);
        factor_s = Translator(student_outputs[2]);

        loss = BETA * (criterion(FT(factor_s), FT(factor_t.detach()))) \
               + criterion_CE(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()
        translator_optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Loss: %.3f | Acc net: %.3f%%|' % (train_loss / (b_idx + 1), 100. * correct / total))

    scheduler.step()
    translator_scheduler.step()

    eval()












