import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

import copy
import numpy as np
import random
from tqdm import trange
from scipy import signal

from utils.options import args_parser
from utils.sampling import noniid
from utils.dataset import load_data
from utils.test import test_img
from src.aggregation import server_opt
from src.update import EdgeOpt

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.tsboard:
        writer = SummaryWriter(f'runs/{args.dataset}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_train, dataset_test = load_data(args)

    # early stopping hyperparameters
    cnt = 0
    check_acc = 0

    # sample users
    dict_users, o_classes = noniid(dataset_train, args)
    net_glob = resnet18(num_classes = args.num_classes).to(args.device)
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    net_glob.train()

    # cyclic server learning rate
    time = np.arange(0, 2, 0.01)
    freq = args.freq
    max_clr = 0
    clr_cycle = args.amp * signal.sawtooth(2 * np.pi * freq * time)
    clr = 1.0 - clr_cycle

    # copy weights
    w_glob = net_glob.state_dict()

    for iter in trange(args.global_ep):
        w_locals = []
        selected_clients = max(int(args.frac * args.num_clients), 1)
        idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

        for idx in idxs_users:
            local = EdgeOpt(args = args, dataset = dataset_train, idxs = dict_users[idx], user_classes = o_classes[idx])
            w = local.train(previous_net = None, global_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))

        # update global weights
        w_glob = server_opt(w_locals, args, clr[iter])

         # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args)

        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")
            print(f"Check accuracy: {check_acc}")
            print(f"patience: {cnt}")

        if check_acc == 0:
            check_acc = test_acc
        elif test_acc < check_acc + args.delta:
            cnt += 1
        else:
            check_acc = test_acc
            cnt = 0

        # early stopping
        if cnt == args.patience:
            print('Early stopped federated training!')
            break

        # tensorboard
        if args.tsboard:
            writer.add_scalar(f'testacc/{args.dataset}_{args.method}_cyclic_{args.cyclic}_amp_{args.amp}_freq_{args.freq}', test_acc, iter)
            writer.add_scalar(f'testloss/{args.dataset}_{args.method}_cyclic_{args.cyclic}_amp_{args.amp}_freq_{args.freq}', test_loss, iter)

    if args.tsboard:
        writer.close()

