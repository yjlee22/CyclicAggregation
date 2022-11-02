import os

cpath = os.getcwd()

folderpath = os.chdir(cpath + '/result/testacc')

def csv_reader(f):
    csv_file = open(f, 'r')
    
    for num in csv_file:
        data = list(map(float, num.split(',')))
            
    return data
    

def performance_v1(dataset, method):
    max_testacc = 0
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if origin in f:
                data = csv_reader(f)
                print(f'{origin} : {max(data)}')
                break
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if cyclic in f:
                data = csv_reader(f)
                if max(data) >= max_testacc:
                    max_testacc = max(data)
    
    print(f'{cyclic} : {max_testacc}')
    
    return

def performance_v2(dataset, method):
    max_testacc = 0
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if origin in f:
            data = csv_reader(f)
            print(f'{origin} : {max(data)}')
            break
    
    for f in os.listdir(folderpath):
        if cyclic in f:
            data = csv_reader(f)
            if max(data) >= max_testacc:
                max_testacc = max(data)
    
    print(f'{cyclic} : {max_testacc}')
    
    return


performance_v1('mnist', 'fedavg')
performance_v1('mnist', 'fedprox')
performance_v1('mnist', 'moon')
performance_v1('mnist', 'fedrs')

performance_v2('fmnist', 'fedavg')
performance_v2('fmnist', 'fedprox')
performance_v2('fmnist', 'moon')
performance_v2('fmnist', 'fedrs')

performance_v2('cifar', 'fedavg')
performance_v2('cifar', 'fedprox')
performance_v2('cifar', 'moon')
performance_v2('cifar', 'fedrs')

performance_v2('svhn', 'fedavg')
performance_v2('svhn', 'fedprox')
performance_v2('svhn', 'moon')
performance_v2('svhn', 'fedrs')