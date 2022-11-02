import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated learning arguments
    parser.add_argument('--method', type=str, default='fedavg', help="aggregation method")
    parser.add_argument('--global_ep', type=int, default=200, help="total number of communication rounds")
    parser.add_argument('--alpha', default=1, type=float, help="random distribution fraction alpha")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--num_data', type=int, default=100, help="number of data per client for label skew")
    parser.add_argument('--frac', type=float, default=0.1, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs: E")
    parser.add_argument('--min_le', type=int, default=1, help="minimum number of local epoch")
    parser.add_argument('--max_le', type=int, default=5, help="maximum number of minimum local epoch")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="client learning rate")
    parser.add_argument('--past', action='store_true', help='utilize past global model')
    parser.add_argument('--score', type=str, default='euclid', help="scoring method")   

    # fedprox, fedrs, and moon arguments
    parser.add_argument('--mu', type=float, default=1e-2, help='hyper parameter for fedprox')
    parser.add_argument('--moon_mu', type=float, default=0.1, help='hyper parameter for moon')
    parser.add_argument('--fedrs_alpha', type=float, default=0.5, help='hyper parameter for fedrs')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--sampling_classes', type=int, default=4, help="number of classes for sampling")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--earlystop', action='store_true', help='early stopping option')
    parser.add_argument('--patience', type=int, default=10, help="hyperparameter of early stopping")
    parser.add_argument('--delta', type=float, default=0.01, help="hyperparameter of early stopping")
    parser.add_argument('--pretrain', action='store_true', help='pretraining option')

    # cyclic aggregation arguments
    parser.add_argument('--cyclic', action='store_true', help='utilize cyclic aggregation')
    parser.add_argument('--freq', type=float, default=0.0, help="number of cycles")
    parser.add_argument('--amp', type=float, default=0.00, help="amplitude parameter")
    
    args = parser.parse_args()
    
    return args
