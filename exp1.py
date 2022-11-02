import os

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
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if origin in f:
                check = csv_reader(f, target)
                print(f'{origin} : {len(check)}')
                break
    
    for f in os.listdir(folderpath):
        if 'fmnist' not in f:
            if cyclic in f:
                check = csv_reader(f, target)
                if len(check) <= min_ge:
                    min_ge = len(check)
    
    print(f'{cyclic} : {min_ge}')
    
    return

def convergence_v2(dataset, method, target):
    min_ge = 200
    tag = ''
    origin = f'{dataset}_{method}_cyclic_False'
    cyclic = f'{dataset}_{method}_cyclic_True'
    
    for f in os.listdir(folderpath):
        if origin in f:
            check = csv_reader(f, target)
            print(f'{origin} : {len(check)}')
            break
    
    for f in os.listdir(folderpath):
        if cyclic in f:
            check = csv_reader(f, target)
            if len(check) <= min_ge:
                min_ge = len(check)
    
    print(f'{cyclic} : {min_ge}')
    
    return

convergence_v1('mnist', 'fedavg', 98)
convergence_v1('mnist', 'fedprox', 98)
convergence_v1('mnist', 'moon', 98)
convergence_v1('mnist', 'fedrs', 98)

convergence_v2('fmnist', 'fedavg', 85)
convergence_v2('fmnist', 'fedprox', 85)
convergence_v2('fmnist', 'moon', 85)
convergence_v2('fmnist', 'fedrs', 85)

convergence_v2('cifar', 'fedavg', 55)
convergence_v2('cifar', 'fedprox', 55)
convergence_v2('cifar', 'moon', 55)
convergence_v2('cifar', 'fedrs', 55)

convergence_v2('svhn', 'fedavg', 81)
convergence_v2('svhn', 'fedprox', 81)
convergence_v2('svhn', 'moon', 81)
convergence_v2('svhn', 'fedrs', 81)
