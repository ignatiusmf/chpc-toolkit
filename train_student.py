from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.student_loader import StudentLoader
from toolbox.loss_functions import Vanilla, Logits_KD
from toolbox.data_loader import DataHelper



data = DataHelper('cifar100')
epochs = 150
teacher = ResNet112
student = ResNet56
loss_funcitons = [
    Vanilla,
    Logits_KD
]



teacher = teacher(data.class_num).to('cuda')
def helper(loss_function):
    return StudentLoader(student, teacher, loss_function(), data, epochs)
models = [helper(l) for l in loss_funcitons]




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



# print("Teacher test loss / acc:") # TODO 
# Actually, make it print a nice table type of thing at then end of training
# Compareing the Teacher loss and accuracy to the trained student loss and accuracy

print("Training started")
for epoch in range(epochs):
    print(f'{epoch=}')
    for model in models:
        model.iterate()

    # plot()
    # save_to_dict() TODO


    # if test_accuracy > max_acc: 
    #     max_acc = test_accuracy
    #     checkpoint = {
    #         'model_state_dict': Student_vanilla.state_dict(),
    #     }
    #     torch.save(checkpoint, f"checkpoint/Student_{epoch}_{test_accuracy:.0f}.pth")




