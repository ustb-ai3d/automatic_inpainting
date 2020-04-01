import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from .config import Config
from .edge_connect import EdgeConnect


def inpainting(mode=None, conf={}):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode, conf=conf)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = EdgeConnect(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None, conf={}):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config_path = os.path.join(conf['checkpoints'], 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(conf['checkpoints']):
        os.makedirs(conf['checkpoints'])

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('models/edge_connect/config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if conf['model']:
            config.MODEL = conf['model']

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = conf['model'] if conf['model'] is not None else 3
        config.INPUT_SIZE = 0

        if conf['input'] is not None:
            config.TEST_FLIST = conf['input']

        if conf['mask'] is not None:
            config.TEST_MASK_FLIST = conf['mask']

        if conf['edge'] is not None:
            config.TEST_EDGE_FLIST = conf['edge']

        if conf['output'] is not None:
            config.RESULTS = conf['output']

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = conf['model'] if conf['model'] is not None else 3

    return config


if __name__ == "__main__":
    main()
