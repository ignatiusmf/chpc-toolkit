import torch
import numpy as np
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(np.array([0,1,2,3,4,5,6]))
