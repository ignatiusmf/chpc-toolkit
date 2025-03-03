import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
device = 'cuda'

def get_path(experiment_name):
    experiment_folder = Path(f'experiments/{experiment_name}')
    experiment_folder.mkdir(parents=True, exist_ok=True)
    existing_folders = [int(f.name) for f in experiment_folder.iterdir() if f.is_dir() and f.name.isdigit()]
    max_folder_num = max(existing_folders, default=0)  # Default to 0 if no folders exist
    if max_folder_num > 0:
        last_folder = experiment_folder / str(max_folder_num)
        complete_file = last_folder / "complete.txt"
        if complete_file.exists():
            max_folder_num += 1
            (experiment_folder / str(max_folder_num)).mkdir()
    else:
        max_folder_num = 1
        (experiment_folder / str(max_folder_num)).mkdir()
    path = experiment_folder / str(max_folder_num)
    print(f"Using path: {path}")
    return max_folder_num, path


class ModelLoader:
    def __init__(self, student, teacher, distillation, data, epochs):
        self.data = data()
        self.distillation = distillation()

        self.student = student(self.data.class_num).to(device)
        if teacher:
            self.teacher = teacher(self.data.class_num).to(device)
            self.name = f'{self.data.name}_{self.teacher.model_type}_{self.student.model_type}_{self.distillation.name}_{epochs}'
            checkpoint = torch.load(f'models/{self.data.name}_{self.teacher.model_type}.pth', weights_only=True)
            self.teacher.load_state_dict(checkpoint['model_state_dict']) # TODO Change to 'weights'
        else:
            assert self.distillation.name == 'vanilla'
            self.teacher = None
            self.name = f'{self.data.name}_{self.student.model_type}_{epochs}'

        self.optimizer = optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        self.max_acc = 0.0

        self.run, self.path = get_path(self.name) 

    def train(self):
        if self.teacher:
            self.teacher.eval()
        self.student.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, targets in self.data.trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()

            outputs = self.student(inputs)

            with torch.no_grad():
                outputs_teacher = self.teacher(inputs) if self.teacher else None

            loss = self.distillation.loss_function(outputs, outputs_teacher, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs[3].data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()

        avg_loss = running_loss / len(self.data.trainloader)
        accuracy = 100 * correct / total

        print(f'TRAIN | Loss: {avg_loss:.3f} | Acc: {accuracy:.2f} |')
        self.scheduler.step()
        self.train_loss.append(avg_loss)
        self.train_acc.append(accuracy)


    def test(self):
        self.student.eval()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, targets in self.data.testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.student(inputs)
            loss = F.cross_entropy(outputs[3], targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs[3].data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()
        avg_loss = running_loss / len(self.data.testloader)
        accuracy = 100 * correct / total

        print(f'TEST | Loss: {avg_loss:.3f} | Acc: {accuracy:.2f} |')
        self.test_loss.append(avg_loss)
        self.test_acc.append(accuracy)

        if accuracy > self.max_acc:
            self.max_acc = accuracy
            self.checkpoint()


    def plot(self):

        plt.plot(np.log10(np.array(self.train_loss)), linestyle='dotted',color='b', label=f'Train Loss')
        plt.plot(np.log10(np.array(self.test_loss)), linestyle='solid',color='b', label=f'Test Loss')

        plt.title(f'{self.name}_{self.run}_Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log10 Loss')
        plt.legend()
        plt.savefig(f'{self.path}/Loss.png')
        plt.close()


        plt.plot(np.array(self.train_acc), linestyle='dotted',color='r', label=f'Train Accuracy')
        plt.plot(np.array(self.test_acc), linestyle='solid',color='r', label=f'Test Accuracy')

        plt.title(f'{self.name}_{self.run}_Accuracy')

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 105, 5))
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.axhline(y=self.max_acc, color='black', linestyle='-', linewidth=0.5)
        plt.text(0, self.max_acc + 1, f"Max Acc = {self.max_acc}", color='black', fontsize=8)


        plt.legend()
        plt.savefig(f'{self.path}/Accuracy.png')
        plt.close()

    def checkpoint(self):
        checkpoint = {
        'weights': self.student.state_dict()
        }
        torch.save(checkpoint, f'{self.path}/weights.pth')

    def log(self):
        logs = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'test_loss': self.test_loss,
            'test_acc': self.test_acc
        }

        with open(f'{self.path}/logs.pkl', 'wb') as f:
            pickle.dump(logs, f)

    def complete(self):
        self.log()
        file_path = Path(f'{self.path}/complete.txt')  # Change to your desired location
        file_path.touch()

# print("Teacher test loss / acc:") # TODO 
# Actually, make it print a nice table type of thing at then end of training
# Compareing the Teacher loss and accuracy to the trained student loss and accuracy