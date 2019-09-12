import os
import sys
import torch
import numpy as np
from datetime import datetime

"""
How To:
Example for running from command line:
    python <path_to>/ProvablyPowerfulGraphNetworks/main_scripts/main_qm9_experiment.py --config=configs/qm9_config.json
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, dataset_name='QM9')

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    torch.manual_seed(100)
    np.random.seed(100)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    if config.target_param is not False:  # (0 == False) while (0 is not False)
        print("target parameter: {0}".format(config.target_param))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)

    data = DataGenerator(config)
    # create an instance of the model you want
    model_wrapper = ModelWrapper(config, data)
    # create trainer and pass all the previous components to it
    trainer = Trainer(model_wrapper, data, config)
    # here you train your model
    trainer.train()
    # test model, restore best model
    test_dists, test_loss = trainer.test(load_best_model=True)

    doc_utils.summary_qm9_results(config.summary_dir, test_dists, test_loss, trainer.best_epoch)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Runtime: {}'.format(datetime.now() - start))