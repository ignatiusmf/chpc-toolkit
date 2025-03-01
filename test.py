import pickle

# Define the file path
file_path = "experiments/Cifar100_ResNetBaby_ResNetBaby_vanilla_150/2/logs.pkl"  # Change to your actual file path

with open(file_path, 'rb') as f:
    logs = pickle.load(f)

print(logs)
