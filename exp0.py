import os
import copy

cpath = os.getcwd()

folderpath = os.chdir(cpath + '/result/testacc')

def csv_reader(f, target):
    csv_file = open(f, 'r')
    
    for num in csv_file:
        data = list(map(float, num.split(',')))
    check = [d for d in data if d <= target]
    
    return check
    

def convergence_v1(dataset, method, target):
    min_ge = 200
    tag = ''
    save = list()
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if origin in f:
                check = csv_reader(f, target)
                print(f'{origin} : {len(check)}')
                print(check)
                break
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if cyclic in f:
                check = csv_reader(f, target)
                if len(check) <= min_ge:
                    min_ge = len(check)
                    tag = str(f)
                    save = copy.deepcopy(check)
    
    print(f'{tag} : {min_ge}')
    print(save)
    
    return

def convergence_v2(dataset, method, target):
    min_ge = 200
    tag = ''
    save = list()
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if origin in f:
            check = csv_reader(f, target)
            print(f'{origin} : {len(check)}')
            print(check)
            break
    
    for f in os.listdir(folderpath):
        if cyclic in f:
            check = csv_reader(f, target)
            if len(check) <= min_ge:
                min_ge = len(check)
                tag = str(f)
                save = copy.deepcopy(check)

    print(f'{tag} : {min_ge}')
    print(save)
    
    return

convergence_v1('mnist', 'fedavg', 98)
convergence_v2('fmnist', 'fedavg', 85)
convergence_v2('cifar', 'fedavg', 55)
convergence_v2('svhn', 'fedavg', 81)