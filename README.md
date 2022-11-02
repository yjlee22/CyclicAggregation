# Fast-Convergent Federated Learning via Cyclic Aggregation 

This is an official implementation of the following paper:
> Youngjoon Lee, Sangwoo Park, and Joonhyuk Kang.
**[Fast-Convergent Federated Learning via Cyclic Aggregation](https://arxiv.org/abs/2210.16520)**  
_submitted to ICASSP 2023_.

## Requirements
The implementation runs on

```bash docker.sh```

Additionally, please install the required packages as below

```pip install tensorboard scipy```

## Federated Learning Techniques
This paper considers the following federated learning techniques
- FedAvg ([McMahan, Brendan, et al. AISTATS 2017](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com))
- FedProx ([Li, Tian, et al. MLSys 2020](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html))
- MOON ([Li, Qinbin, Bingsheng He, and Dawn Song. CVPR 2021](https://arxiv.org/abs/2103.16257))
- FedRS ([Li, Xin-Chun, and De-Chuan Zhan. KDD 2021](https://dl.acm.org/doi/abs/10.1145/3447548.3467254?casa_token=5VXRZ5kg5a4AAAAA:Ll6o5SjATYoZySExzPQp2ioBat7dBtaLUeg9oqu1nqd_zYx-iL9FnZHI4aFOY9tNpQpWrPWHn83JfjI))

## Datasets
- MNIST
- FMNIST
- CIFAR-10
- SVHN

## Usage    
Here is an example to run with cyclic learning rate on MNIST

```python main.py --gpu 0 --tsboard --method fedavg --dataset mnist --cyclic --amp 0.01 --freq 5```

## Acknowledgements
Referred http://doi.org/10.5281/zenodo.4321561
