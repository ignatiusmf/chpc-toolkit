import numpy as np
import matplotlib.pyplot as plt
import json



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