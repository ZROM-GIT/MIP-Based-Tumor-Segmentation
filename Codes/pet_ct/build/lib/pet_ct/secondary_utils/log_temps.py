import psutil
import subprocess
import time
import json
from datetime import datetime
from pathlib2 import Path

def get_cpu_temp():
    temp = psutil.sensors_temperatures()
    if 'coretemp' in temp:
        return temp['coretemp'][0].current
    elif 'cpu_thermal' in temp:
        return temp['cpu_thermal'][0].current
    else:
        return None

def get_gpu_temps():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    temps = result.stdout.decode().splitlines()  # Decode and split into lines
    return [int(temp) for temp in temps]  # Convert each line to an integer

def log_temps(epoch, batch, set_name, args):
    if_log_temps = getattr(args, 'log_temps', False)
    if not if_log_temps:
        return
    cpu_temp = get_cpu_temp()
    gpu_temps = get_gpu_temps()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Name = f'{args.project_name}{args.experiment_number}_{args.experiment_name}'
    data = {
        "time": now,
        "set": set_name,
        "epoch": epoch,
        "batch": batch,
        "gpu_temps": gpu_temps,
        "cpu_temp": cpu_temp
    }
    path = Path(f'{args.project_dir}/Logs/Temperatures/{Name}.json')
    path.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as file:
        file.write(json.dumps(data) + '\n')  # Log data as JSON line-by-line

