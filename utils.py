import os
import re
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
choice_param_name = ['alpha', 'beta', 'gamma']
lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct']
init_constrain = 0.2
def randomize_gate(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = randomize_gate(module)
        if all([hasattr(module, i) for i in choice_param_name]):
            torch.nn.init.uniform_(module.alpha, a=-(0.5 * init_constrain), b=(0.5 * init_constrain))
            torch.nn.init.uniform_(module.beta, a=-(0.5 * init_constrain), b=(0.5 * init_constrain))
            torch.nn.init.uniform_(module.gamma, a=-(0.5 * init_constrain), b=(0.5 * init_constrain))
    return model

def deletStrmodule(checkpoint: dict):
    outerkey = list(checkpoint.keys())
    new_dict = {}
    new_dict[outerkey[0]] = OrderedDict()
    for k, v in checkpoint[outerkey[0]].items():
        name = k[7:]
        new_dict[outerkey[0]][name] = v
    return new_dict

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, epoch, tag=''):
    if not os.path.exists("./raw/models"):
        os.makedirs("./raw/models")
    filename = os.path.join(
        "./raw/models/{}-checkpoint-{:06}.pth.tar".format(tag, epoch))
    torch.save(state, filename)

def get_model(modeltag, addr):
    if addr is not None:
        model_list = os.listdir(addr)
    else:
        return None, 0
    if model_list == []:
        return None, 0
    model_list.sort()
    cand_model = []
    for m in model_list:
        if modeltag in m:
            cand_model.append(m)
    lastest_model = cand_model[-1]
    print('The model checkpoint matching the provided modeltag is \t', lastest_model)
    return addr + '/' + lastest_model

def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups

def record_param(args, model, dict, epoch, modeltag, store=False):
    #store the dict
    if dict is None:
        return None
    if store:
        if not os.path.exists('./dicts_for_params'):
            os.mkdir('./dicts_for_params')
        np.save(os.path.join('./dicts_for_params', modeltag + '.npy'), dict)
    # elif (args.imagenet or args.fashion_mnist or args.mnist or args.cifar100):
    else:
        for pname, p in model.named_parameters():
            n = pname.split('.')
            if n[-1] in choice_param_name + lifcal_param_name:
                if len(n) < 4:
                    continue
                num_list = list(map(int, re.findall(r"\d+", pname)))
                if len(num_list) > 1:
                    layer = int(num_list[0]) * 2
                else:
                    layer = int(num_list[0]) * 2 + 1
                dict[n[-1]][layer].append(p.clone().detach().cpu().numpy())

def read_param(epoch, modeltag):
    if not os.path.exists(os.path.join('./dicts_for_params', modeltag + '.npy')):
        print('no checkpoint found, skip reading')
        return None
    a = np.load(os.path.join('./dicts_for_params', modeltag + '.npy'), allow_pickle = True).item()

    return a

def create_para_dict(args, model):
    # create dict
    para_dict = {}
    layer = None
    for pname, p in model.named_parameters():
        n = pname.split('.')
        if n[-1] in choice_param_name + lifcal_param_name:
            if len(n) < 4:
                continue
            num_list = list(map(int, re.findall(r"\d+", pname)))
            if len(num_list) > 1:
                layer = int(num_list[0]) * 2
            else:
                layer = int(num_list[0]) * 2 + 1
            para_dict[n[-1]] = []
    if layer is not None:
        for key in para_dict.keys():
            para_dict[key] = [[] for i in range(layer + 1)]
    else:
        return None

    return para_dict
