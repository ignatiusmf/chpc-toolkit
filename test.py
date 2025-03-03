from pathlib import Path
def get_path(experiment_name, ):
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


get_path('Cifar100_ResNet112')