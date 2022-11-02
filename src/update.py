import copy
from random import randint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import DatasetSplit

def restricted_softmax(logits, o_classes, args):
    m_logits = torch.ones_like(logits[0]).to(args.device) * args.fedrs_alpha
    
    for c in o_classes:
        m_logits[c] = 1.0
    
    for i in range(len(logits)):
        logits[i] = torch.mul(logits[i], m_logits)

    return logits
        
class EdgeOpt(object):
    def __init__(self, args, dataset=None, idxs=None, user_classes=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.user_classes = user_classes
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def train(self, previous_net, global_net, net):
        
        local_ep = randint(self.args.min_le, self.args.max_le)

        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)

        for iter in range(local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)

                #fedrs: https://dl.acm.org/doi/10.1145/3447548.3467254
                if self.args.method == 'fedrs':
                    log_probs = restricted_softmax(log_probs, self.user_classes, self.args)
                
                loss = self.loss_func(log_probs, labels.squeeze(dim=-1))

                # fedprox: https://arxiv.org/abs/1812.06127
                if self.args.method == 'fedprox':
                    if iter > 0: 
                        for w, w_t in zip(global_net.parameters(), net.parameters()):
                            loss += self.args.mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)
                            w_t.grad.data += self.args.mu * (w_t.data - w.data)

                # moon: https://arxiv.org/abs/2103.16257
                if self.args.method == 'moon':
                    pro1 = nn.Sequential(*list(net.children())[:-1])(images).squeeze()
                    pro2 = nn.Sequential(*list(global_net.children())[:-1])(images).squeeze()
                    pro3 = nn.Sequential(*list(previous_net.children())[:-1])(images).squeeze()
                
                    posi = self.cos(pro1, pro2)
                    logits = posi.reshape(-1,1)
                
                    nega = self.cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=-1)
                    logits /= 0.5
                    loss += self.args.moon_mu * self.loss_func(logits, torch.zeros(images.size(0)).cuda().long().to(logits.device))
            
                loss.backward()
                
                optimizer.step()

        return net.state_dict()