from pathlib import Path


def get_names(data_name, student_name, teacher_name=None, distillation_name=None, experiment_id=None, scan=False):
    if teacher_name and distillation_name:
        experiment_name = f'{data_name}/{student_name}_{teacher_name}/{distillation_name}'
    else:
        assert not teacher_name
        assert not distillation_name
        experiment_name = f'{data_name}/{student_name}'
    model_name = f'{data_name}_{student_name}'
    if experiment_id:
        experiment_folder = Path(f'experiments/{experiment_name}/{experiment_id}')
        experiment_folder.mkdir(parents=True, exist_ok=True)
        path = experiment_folder
        print(f"Using path: {path}")
        return experiment_name, experiment_id, model_name, path
    else:
        experiment_folder = Path(f'experiments/{experiment_name}')
        experiment_folder.mkdir(parents=True, exist_ok=True)
        existing_folders = [int(f.name) for f in experiment_folder.iterdir() if f.is_dir() and f.name.isdigit()]
        max_folder_num = max(existing_folders, default=0)  
        if max_folder_num > 0:
            max_folder_num += 1
        else:
            max_folder_num = 1
        (experiment_folder / str(max_folder_num)).mkdir()
        path = experiment_folder / str(max_folder_num)
        print(f"Using path: {path}")
        return experiment_name, max_folder_num, model_name, path


import matplotlib.pyplot as plt
import numpy as np
def plot_the_things(train_loss, test_loss, train_acc, test_acc, name, run, path):
        plt.plot(np.log10(np.array(train_loss)), linestyle='dotted',color='b', label=f'Train Loss')
        plt.plot(np.log10(np.array(test_loss)), linestyle='solid',color='b', label=f'Test Loss')

        if run.isdigit():
            plt.title(f'{name}_Loss')
        else:
            plt.title(f'{name}_{run}_Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log10 Loss')
        plt.legend()
        plt.savefig(f'{path}/Loss.png')
        plt.close()

        max_acc = np.max(np.array(test_acc))

        plt.plot(np.array(train_acc), linestyle='dotted',color='r', label=f'Train Accuracy')
        plt.plot(np.array(test_acc), linestyle='solid',color='r', label=f'Test Accuracy')

        if run.isdigit():
            plt.title(f'{name}_Accuracy')
        else:
            plt.title(f'{name}_{run}_Accuracy')

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 105, 5))
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.axhline(y=max_acc, color='black', linestyle='-', linewidth=0.5)
        plt.text(0, max_acc + 1, f"Max Acc = {max_acc}", color='black', fontsize=8)


        plt.legend()
        plt.savefig(f'{path}/Accuracy.png')
        plt.close()

import torch.nn.functional as F
import torch

def evaluate_model(model, loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[3], targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    print(f'TEST | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    return val_loss/(b_idx+1), correct*100/total



import argparse
from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from sandbox.toolbox.data_loader import Cifar10, Cifar100

DATASETS = {
    'Cifar10': Cifar10,
    'Cifar100': Cifar100
}

MODELS = {
    'ResNet112': ResNet112,
    'ResNet56': ResNet56,
    'ResNet20': ResNet20,
    'ResNetBaby': ResNetBaby,
    'None': None
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run a training script with custom parameters.')
    
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--data', type=str, default='Cifar100', choices=DATASETS.keys())
    parser.add_argument('--student', type=str, default='ResNet56', choices=MODELS.keys())
    parser.add_argument('--teacher', type=str, default='None', choices=MODELS.keys())
    parser.add_argument('--experiment-id', type=str, default=None)
    
    args = parser.parse_args()
    return args

from pprint import pprint
def get_settings():
    args = parse_args()
    settings = {
        'Epochs': args.epochs,
        'Data': DATASETS[args.data], 
        'Student': MODELS[args.student],  
        'Teacher': MODELS[args.teacher], 
        'experiment_id': args.experiment_id
    }
    pprint(settings)
    return settings