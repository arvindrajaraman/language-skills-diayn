from datetime import datetime
import subprocess
import sys

def get_optimal_qos():
    output = subprocess.getoutput('squeue')
    curr_jobs = output.count('arvind')
    if curr_jobs < 2:
        return 'high'
    elif curr_jobs < 3:
        return 'medium'
    elif curr_jobs < 5:
        return 'default'
    else:
        return 'scavenger'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <command>'.format(sys.argv[0]))
        sys.exit(1)

    command = ' '.join(sys.argv[1:])
    timestamp = int(datetime.now().timestamp())
    args_dict = {
        'job-name': 'll3_sweep',
        'cpus-per-task': '1',
        'time': '24:00:00',
        'gpus': 1,
        'cpus-per-task': 3,

        'mem': '4gb',
        'qos': get_optimal_qos(),
        'output': f'slurm_logs/{timestamp}-out.txt',
        'error': f'slurm_logs/{timestamp}-err.txt'
    }

    args = ''
    for k, v in args_dict.items():
        args += f'--{k}={v} '

    full_command = 'srun ' + args + 'conda run --no-capture-output -n diayn ' + command
    print(f'Logging at: slurm_logs/{timestamp}-out.txt and slurm_logs/{timestamp}-err.txt')
    print('Running:', full_command)
    subprocess.run(full_command, shell=True)

# python slurm.py python sweep_run.py -i haovjuq1
