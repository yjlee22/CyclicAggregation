from torchvision import datasets, transforms
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_data(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=trans_fmnist)

    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('../data/svhn/train/', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('../data/svhn/test/', split='test', download=True, transform=trans_svhn)
    
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized dataset')
    
    return dataset_train, dataset_test