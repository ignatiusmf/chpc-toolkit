import numpy as np
import matplotlib.pyplot as plt
import json


def bigloss():
    with open('checkpoint/bigloss.json', 'r') as f:
        loss_data = json.load(f)


    smoothing = 782 

    # lossi = np.power(10, np.array(loss_data))
    lossi = np.array(loss_data)
    LargeCNN = np.convolve(lossi[:,0], np.ones(smoothing)/smoothing, mode='valid') 
    SmallCNN = np.convolve(lossi[:,1], np.ones(smoothing)/smoothing, mode='valid')
    plt.plot(LargeCNN, label="LargeCNN")
    plt.plot(SmallCNN, label="SmallCNN")
    # plt.plot(lossi[:,0], label="LargeCNN")
    # plt.plot(lossi[:,1], label="SmallCNN")

    for x in range(0, len(lossi[:,0]), 782):
        plt.axvline(x=x, color='gray', linestyle='--',linewidth=0.5)

    plt.xlabel("Batch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.show()


with open('checkpoint/test_acc.json', 'r') as f:
    test_acc = np.array(json.load(f))

with open('checkpoint/KD_test_acc.json', 'r') as f:
    kd_test_acc = np.array(json.load(f))



plt.plot(test_acc[:,0], label="LargeCNN")
plt.plot(test_acc[:,1], label="SmallCNN")
plt.plot(kd_test_acc[:,0], label="Student")
plt.plot(kd_test_acc[:,1], label="Control")

plt.ylim(0,100)
plt.xlabel("epoch")
plt.ylabel("Test score")
plt.legend()
plt.show()