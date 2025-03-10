import subprocess
import os
from sandbox.toolbox.utils import get_names

def generate_pbs_script(
    walltime="02:00:00",
    epochs=150,
    data="cifar100",
    student="resnet56",
    teacher=None,
    distillation=None,
    experiment_id=None
):

    if distillation:
        python_file_path = f'/mnt/lustre/users/iferreira/chpc-toolkit/train_{distillation}.py' 
    else:
        python_file_path = f'/mnt/lustre/users/iferreira/chpc-toolkit/train.py' 
    
    experiment_name, experiment_id, _, path = get_names(data,student,teacher,distillation,experiment_id)


    python_cmd = f"python {python_file_path}"
    python_cmd += f" --epochs {epochs}"
    python_cmd += f" --data {data}"
    python_cmd += f" --student {student}"
    if teacher is not None:
        python_cmd += f" --teacher {teacher}"
    if experiment_id is not None:  
        python_cmd += f" --experiment-id {experiment_id}"

    print(python_cmd)

    new_path = str(path).replace("\\","/")
    output_log = f'/mnt/lustre/users/iferreira/chpc-toolkit/{new_path}/logs'
    error_log = f'/mnt/lustre/users/iferreira/chpc-toolkit/{new_path}/errors'

    job_name = f'{experiment_name.replace("/","_")}_{experiment_id}'

    pbs_script = f"""#!/bin/sh
#PBS -N {job_name}
#PBS -q gpu_1
#PBS -P CSCI1166
#PBS -l select=1:ncpus=10:mpiprocs=10:mem=32gb:ngpus=1
#PBS -l walltime={walltime}
#PBS -o {output_log}
#PBS -e {error_log}
#PBS -m abe -M u25755422@tuks.co.za

ulimit -s unlimited
module load chpc/python/anaconda/3-2021.11
source /mnt/lustre/users/iferreira/myenv/bin/activate

date
echo -e 'Running {python_cmd}\\n'

start_time=$(date +%s) 

cd /mnt/lustre/users/iferreira/chpc-toolkit
{python_cmd}

echo -e "\\nTotal execution time: $(( $(date +%s) - start_time)) seconds"
"""
    
    temp_file = "temp_pbs_script.sh"
    with open(temp_file, 'w') as f:
        f.write(pbs_script)
    try:
        result = subprocess.run(['qsub', temp_file], capture_output=True, text=True)
        print(f"Job submitted: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

generate_pbs_script(
    data='Cifar100',
    student='ResNet56',
    teacher='ResNet112',
    distillation='kd' 
)
