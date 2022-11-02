import numpy as np

def noniid(dataset, args):
    idxs = np.arange(len(dataset))
    
    if args.dataset == 'mnist':
        labels = dataset.targets.numpy()
    elif args.dataset == 'fmnist':
        labels = dataset.targets.numpy()
    elif args.dataset == 'cifar':
        labels = np.array(dataset.targets)
    elif args.dataset == 'svhn':
        labels = np.array(dataset.labels)
    else:
        exit('Error: unrecognized dataset')

    dict_users = {i: list() for i in range(args.num_clients)}
    dict_labels = dict()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = list(idxs_labels[0])
    labels = idxs_labels[1]
    
    rand_class_num = np.random.randint(0, args.num_classes, size= (args.num_clients, args.sampling_classes))

    for i in range(args.num_classes):
        specific_class = set(np.extract(labels == i, idxs))
        dict_labels.update({i : specific_class})

    for i, class_num in enumerate(rand_class_num):
        rand_set = list()
        
        rand_class1 = list(np.random.choice(list(dict_labels[class_num[0]]), args.num_data))
        rand_class2 = list(np.random.choice(list(dict_labels[class_num[1]]), args.num_data))
        rand_class3 = list(np.random.choice(list(dict_labels[class_num[2]]), args.num_data))
        rand_class4 = list(np.random.choice(list(dict_labels[class_num[3]]), args.num_data))
        
        dict_labels[class_num[0]] = dict_labels[class_num[0]] - set(rand_class1)
        dict_labels[class_num[1]] = dict_labels[class_num[1]] - set(rand_class2)
        dict_labels[class_num[2]] = dict_labels[class_num[2]] - set(rand_class3)
        dict_labels[class_num[3]] = dict_labels[class_num[3]] - set(rand_class4)
 
        rand_set = rand_set + rand_class1 + rand_class2 + rand_class3 + rand_class4
        dict_users[i] = set(rand_set)

    return dict_users, rand_class_num