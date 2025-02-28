import torch
import torch.optim as optim
import torch.nn.functional as F


from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.data_loader import get_loaders

device = 'cuda'

# Assumes you have a pretrained teacher model of the type you specified, 
# named teacher_problem_set_model_type, i.e., teacher_cifar10_ResNet112
class StudentLoader:
    def __init__(self, student, teacher, loss_fn, data, epochs):

        self.teacher = teacher
        self.student = student(data.class_num).to(device)
        self.loss_fn = loss_fn.loss_function
        self.optimizer = optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        self.name = loss_fn.name

        self.trainloader, self.testloader = data.trainloader, data.testloader


        checkpoint = torch.load(f'checkpoint/teacher_{data.name}_{self.teacher.model_type}.pth', weights_only=True)
        self.teacher.load_state_dict(checkpoint['model_state_dict'])

    def train(self):
        self.teacher.eval()
        self.student.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, targets in self.trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()

            outputs = self.student(inputs)

            with torch.no_grad():
                outputs_teacher = self.teacher(inputs)

            loss = self.loss_fn(outputs, outputs_teacher, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs[3].data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()

        avg_loss = running_loss / len(self.trainloader)
        accuracy = 100 * correct / total

        print(f"Loss: {avg_loss:.3f} | Train accuracy: {accuracy:.3f}% |")
        self.scheduler.step()
        self.train_loss.append(avg_loss)
        self.train_acc.append(accuracy)


    def test(self):
        self.student.eval()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, targets in self.testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.student(inputs)
            loss = F.cross_entropy(outputs[3], targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs[3].data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()
        avg_loss = running_loss / len(self.testloader)
        accuracy = 100 * correct / total

        print(f"Loss: {avg_loss:.3f} | Test accuracy: {accuracy:.3f}% |")
        self.test_loss.append(avg_loss)
        self.test_acc.append(accuracy)

    def iterate(self):
        print(f'Student: {self.name}')
        self.train()
        self.test()
        print()
