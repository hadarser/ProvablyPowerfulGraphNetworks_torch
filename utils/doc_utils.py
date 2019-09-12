import matplotlib.pyplot as plt
import pandas as pd
import os
import json


BASE_DIR = os.path.abspath("../experiments")


def write_to_file_doc(train_acc, train_loss, val_acc, val_loss, epoch, config):
    """
    Creates if not exist and update summary csv file of the the training.
    If test_acc or test_loss are None, does not document the test values.
    """
    val = [config.exp_name, epoch, train_loss, train_acc, val_loss, val_acc, config.timestamp]
    columns = ['experiment_name', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'timestamp']

    fullpath = os.path.join(config.summary_dir, 'per_epoch_stats.csv')
    if os.path.exists(fullpath):
        f = pd.read_csv(fullpath)
    else:
        f = pd.DataFrame(columns=columns)
    f = f.append(pd.DataFrame([val], columns=columns))
    f.to_csv(fullpath, index=False)


def create_experiment_results_plot(title, parameter, directory, log=False):
    """
    create plot of chosen parameter during training
    :param title: the first part of plot title
    :param parameter: the parameter you want to plot. loss accuracy
    :param directory: the directory in which the plot will be saves
    :param log: boolean to chose if you want semilog scale
    :return: name of the saved plot file
    """
    fullpath = os.path.join(directory, 'per_epoch_stats.csv')
    df = pd.read_csv(filepath_or_buffer=fullpath)
    epochs = df["epoch"]
    train_param = df["train_{0}".format(parameter)]
    val_param = df["val_{0}".format(parameter)]

    if log:
        plt.semilogy(epochs, train_param, 'r', label='train')
        plt.semilogy(epochs, val_param, 'b', label='validation')
        plt.ylabel(parameter + " semilog")
        axes = plt.gca()
        axes.set_ylim([10 ** -3, 10 ** 1])
    else:
        plt.plot(epochs, train_param, 'r', label='train')
        plt.plot(epochs, val_param, 'b', label='validation')
        plt.ylabel(parameter)
        axes = plt.gca()
        axes.set_ylim([0.1, 0.95])
    plt.xlabel('Epochs')

    plt.title(title)
    plt.legend()
    file_name = (os.path.join(directory, (title + parameter + ".png")))
    plt.savefig(file_name)

    plt.close()
    return file_name


def get_folders_list(directory):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            r.append(os.path.join(directory, name))
    return r


def summary_10fold_results(summary_dir):
    df = pd.read_csv(os.path.join(summary_dir, 'per_epoch_stats.csv'))

    df = df.drop(['train_loss', 'train_accuracy', 'timestamp', 'experiment_name'], axis=1)  # drop irrelevant columns
    df_group_std = df.groupby('epoch').std()
    df_group = df.groupby('epoch').mean()
    df_group['std'] = df_group_std.val_accuracy
    best_epoch = df_group['val_accuracy'].idxmax()

    best_row = df_group.loc[best_epoch]
    print("Results")
    print("Best epoch - {0}".format(best_epoch))
    print("Mean Accuracy = {0}".format(best_row['val_accuracy']))
    print("Mean std = {0}".format(best_row['std']))

    # Document the validation results of the best epoch, per experiment
    df2 = df[df.epoch == best_epoch].copy()
    df2['fold'] = pd.Series(range(10), index=df2.index) + 1
    df2 = df2.append(pd.Series([best_epoch, best_row['val_loss'], best_row['val_accuracy'], 'mean'], index=df2.columns),
                     ignore_index=True)
    fullpath = os.path.join(summary_dir, 'exp_summary.csv')
    df2.to_csv(fullpath, index=False)


def summary_qm9_results(summary_dir, test_dists, test_loss, best_epoch, create_csv=True):
    print("Results")
    print("\tTest absolute distances by best epoch ({0}) of validation = \n{1}".format(best_epoch, test_dists))

    if not create_csv:
        return

    # Write two lines to csv file
    test_dists = list(test_dists)
    if len(test_dists) > 1:
        # 12 targets
        columns = ['test_loss'] + ['dist{}'.format(i) for i in range(1, 13)]
    else:
        columns = ['test_loss', 'dist']
    values1 = [test_loss] + test_dists

    fullpath = os.path.join(summary_dir, 'exp_summary.csv')
    pd.DataFrame([values1], columns=columns).to_csv(fullpath, index=False)


def doc_used_config(config):
    """
    Log the used config file in the summary directory in JSON format
    :param config: configuration EasyDict
    :return:
    """
    fullpath = os.path.join(config.summary_dir, 'used_config.json')

    with open(fullpath, 'w') as fp:
        json.dump(config, fp)


if __name__ == "__main__":
    print(BASE_DIR)
