from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from sandbox.toolbox.data_loader import Cifar10, Cifar100

from sandbox.toolbox.factor_transfer_components import Paraphraser, Translator
from sandbox.toolbox.utils import get_names, plot_the_things, evaluate_model, get_settings

import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda'

################## SPECIFY SETTINGS ######################
settings = get_settings()

Epochs = settings['Epochs']
Data = settings['Data']
Student = settings['Student']
Teacher = settings['Teacher']
expirement_id = settings['experiment_id']

################## INITIALIZING THE THINGS ######################
Data = Data()

trainloader, testloader = Data.trainloader, Data.testloader

Student = Student(Data.class_num).to(device)
Teacher = Teacher(Data.class_num).to(device)
checkpoint = torch.load(f'models/{Data.name}_{Teacher.model_type}.pth', weights_only=True)
Teacher.load_state_dict(checkpoint['weights'])

experiment_name, experiment_id, model_name, path = get_names(Data.name, Student.model_type, Teacher.model_type, 'ft', expirement_id) 

optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
p_loss = []
max_acc = 0

################## FACTOR TRANSFER INITIALIZATION ##################

Paraphraser_Epochs = 25
paraphraser_compression = 0.5

Paraphraser = Paraphraser(64, int(round(64*paraphraser_compression))).to(device)
paraphraser_optimizer = optim.SGD(Paraphraser.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
paraphraser_scheduler = optim.lr_scheduler.CosineAnnealingLR(paraphraser_optimizer, T_max=Paraphraser_Epochs)
criterion = nn.L1Loss()

p_min_loss = 1000 
for epoch in range(Paraphraser_Epochs):
    print(f'{epoch=}')
    Teacher.eval()
    Paraphraser.train()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        paraphraser_optimizer.zero_grad()
        outputs = Teacher(inputs)
        output_p = Paraphraser(outputs[2],0)
        loss = criterion(output_p, outputs[2].detach())
        loss.backward()
        paraphraser_optimizer.step()
        val_loss += loss.item()
        b_idx = batch_idx
    paraphraser_scheduler.step()
    avg_loss = val_loss / (b_idx + 1)
    p_loss.append(avg_loss)
    print(f'Loss: {avg_loss:.3f}')
    if p_min_loss > avg_loss:
        p_min_loss = avg_loss
        torch.save({'weights': Paraphraser.state_dict()}, f'{path}/paraphraser.pth')

    plt.plot(np.log10(np.array(p_loss)), linestyle='dotted',color='b', label=f'Loss')
    plt.title(f'Paraphraser_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log10 Loss')
    plt.legend()
    plt.savefig(f'{path}/Paraphraser_Loss.png')
    plt.close()

################## FACTOR TRANSFER TRAINING ##################

BETA = 500
Translator = Translator(64, int(round(64*paraphraser_compression))).to(device)
translator_optimizer = optim.SGD(Translator.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
translator_scheduler = optim.lr_scheduler.CosineAnnealingLR(translator_optimizer, T_max=Epochs)

def FT(x):
    return F.normalize(x.view(x.size(0), -1))

for epoch in range(Epochs):
    print(f'{epoch=}')
    Teacher.eval()
    Paraphraser.eval()
    Student.train()
    Translator.train()
    val_loss, correct, total = 0, 0, 0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        translator_optimizer.zero_grad()

        ###################################################################################
        teacher_outputs = Teacher(inputs)
        student_outputs = Student(inputs)

        factor_t = Paraphraser(teacher_outputs[2],1);
        factor_s = Translator(student_outputs[2]);

        loss = BETA * (criterion(FT(factor_s), FT(factor_t.detach()))) + F.cross_entropy(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()
        translator_optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    scheduler.step()
    translator_scheduler.step()

    print(f'TRAIN | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    train_loss.append(val_loss/(b_idx+1))
    train_acc.append(correct*100/total)

    tl, ta = evaluate_model(Student, testloader)
    test_loss.append(tl)
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








