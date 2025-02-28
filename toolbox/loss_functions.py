
import torch.nn.functional as F

def vanilla(student, teacher, targets):
    loss = F.cross_entropy(student[3], targets, label_smoothing=0.1)
    return loss

def logits_kd(student, teacher, targets):
    T = 2.0
    alpha = 0.9
    soft_targets = F.kl_div(
        F.log_softmax(student[3] / T, dim=1),
        F.softmax(teacher[3] / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_targets = F.cross_entropy(student[3], targets)
    return alpha * soft_targets + (1 - alpha) * hard_targets

def features_kd():
    print("yeet")

def td_kd():
    print("yeet")

def image_denoise_kd():
    print("yeet")


class LossFunction:
    def __init__(self, loss_function, name):
        self.loss_function = loss_function
        self.name = name

def Vanilla():
    return LossFunction(vanilla, 'vanilla')

def Logits_KD():
    return LossFunction(logits_kd, 'logits_kd')




