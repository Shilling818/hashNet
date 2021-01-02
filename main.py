import torch
from utils.train import trainModel
from utils.data_loader import loadData
from utils.evaluate import match

import argparse
import numpy as np
import random
from loguru import logger
import os


def loadConfig():
    parser = argparse.ArgumentParser(description='DHN_PyTorch')
    parser.add_argument('--dataset_path', default='data', type=str,
                        help='Path of dataset')
    parser.add_argument('--save_checkpoints_path', default='checkpoints/fc', type=str,
                        help='Path of dataset')
    parser.add_argument('--load_checkpoints_path', default=None, type=str,
                        help='Path of dataset')
    parser.add_argument('--arch', default='fc', type=str,
                        help='model name.')
    parser.add_argument('--code-length', default=16, type=int,
                        help='Binary hash code length.')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Batch size.(default: 256)')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--max-iter', default=300, type=int,
                        help='Number of iterations.(default: 500)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--topk', default=20, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--lamda', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=1, type=int,
                        help='Evaluation interval.(default: 10)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


def main(args):

    logger.add('logs/model_{}_code_{}_lamda_{}.log'.format(
        args.arch,
        args.code_length,
        args.lamda,
    ),
        rotation='500 MB',
        level='INFO',
    )

    # set seed
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_dataloader, valid_dataloader, retrieval_dataloader, simMatrix = loadData(
        args.dataset_path,
        args.batch_size,
        args.num_workers,
        args.arch
    )

    if not os.path.exists(args.save_checkpoints_path):
        os.makedirs(args.save_checkpoints_path)

    # training
    trainModel(
        train_dataloader,
        valid_dataloader,
        retrieval_dataloader,
        simMatrix,
        args.code_length,
        args.device,
        args.lr,
        args.max_iter,
        args.lamda,
        args.topk,
        args.evaluate_interval,
        args.arch,
        args.load_checkpoints_path,
        args.save_checkpoints_path
    )


if __name__ == '__main__':
    args = loadConfig()

    # train
    main(args)

    # test
    # test_path = './data/test.mat'
    # retrieval_path = './data/retrieval.mat'
    # model_path = './checkpoints/fc/model_fc_code_16_lamda_1.pt'
    # match(test_path, retrieval_path, args.code_length, model_path, args.device, args.batch_size, args.num_workers,
    #       args.arch)
