from toolbox.model_loader import ModelLoader
from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby
from toolbox.loss_functions import Vanilla, Logits_KD
from toolbox.data_loader import Cifar10, Cifar100



Epochs = 150
Data = Cifar100
Student = ResNetBaby 
Teacher = None
Distillation = Vanilla


model = ModelLoader(Student, Teacher, Distillation, Data, Epochs)


print("Training started")
for epoch in range(Epochs):
    print(f'{model.name} {epoch=}')
    model.train()
    model.test()
    model.plot()

model.complete()



# Need to add parameters to the python script,
# Then make a python file that will run the run.job job and pass the correct parameters to this trainer.py python script 
# And then also specify which directory the standard.out and error.out should be in, want to do it in the 1 2 3 folders, 
# maybe reuse the get_path function in model loader
# Maybe call a seperate python file from within the shell and get the data I need, then after the job completes
# Maybe do so within the shell itself  



# Also, mayb let me specify the run name, instaed of numbers, then I can tweak different things and it won't be in that 1 2 3 4 format

# Also maybe rename the weights that are saved when a model achviese new highest accuracy, currently just "weights"


# What teacher should it load when you specify, e.g., ResNet112? SHould it choose one automatically or should you manually move models to a sepicf placy??



# Time how long an epoch takes and add that to the model object