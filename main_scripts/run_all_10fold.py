import os

datasets = ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC']
run_command = 'python main_10fold_experiment.py '
args1 = '--config=../configs/10fold_config.json '
args2 = '--dataset_name=%s'


def main():
    os.system('pwd')
    for i in range(len(datasets)):
        command = run_command + args1 + args2 % datasets[i]
        os.system(command)


if __name__ == '__main__':
    main()
