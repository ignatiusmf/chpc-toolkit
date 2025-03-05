from pathlib import Path

def get_path(experiment_name, experiment_small_name=None):
    if experiment_small_name:
        experiment_folder = Path(f'experiments/{experiment_name}/{experiment_small_name}')
        experiment_folder.mkdir(parents=True, exist_ok=True)
        path = experiment_folder
        print(f"Using path: {path}")
        return experiment_small_name, path
    else:
        experiment_folder = Path(f'experiments/{experiment_name}')
        experiment_folder.mkdir(parents=True, exist_ok=True)
        existing_folders = [int(f.name) for f in experiment_folder.iterdir() if f.is_dir() and f.name.isdigit()]
        max_folder_num = max(existing_folders, default=0)  # Default to 0 if no folders exist
        if max_folder_num > 0:
            last_folder = experiment_folder / str(max_folder_num)
            complete_file = last_folder / "complete.txt"
            if True: # complete_file.exists():
                max_folder_num += 1
                (experiment_folder / str(max_folder_num)).mkdir()
        else:
            max_folder_num = 1
            (experiment_folder / str(max_folder_num)).mkdir()
        path = experiment_folder / str(max_folder_num)
        print(f"Using path: {path}")
        return max_folder_num, path


import matplotlib.pyplot as plt
import numpy as np
def plot_the_things(train_loss, test_loss, train_acc, test_acc, name, run, path):
        plt.plot(np.log10(np.array(train_loss)), linestyle='dotted',color='b', label=f'Train Loss')
        plt.plot(np.log10(np.array(test_loss)), linestyle='solid',color='b', label=f'Test Loss')

        plt.title(f'{name}_{run}_Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log10 Loss')
        plt.legend()
        plt.savefig(f'{path}/Loss.png')
        plt.close()

        max_acc = np.max(np.array(test_acc))

        plt.plot(np.array(train_acc), linestyle='dotted',color='r', label=f'Train Accuracy')
        plt.plot(np.array(test_acc), linestyle='solid',color='r', label=f'Test Accuracy')

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