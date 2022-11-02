import os
import torch
import random
import collections
import numpy as np
import pandas as pd
from rich.console import Console
from rich.traceback import install
# from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM

install(show_locals=False)
console = Console(record=False, soft_wrap=True)
print = console.print

def set_metaverse_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def str_len(text):
    t = str(text)
    chinese_char = 0
    alphabetic_char = 0
    for c in t:
        if u'\u4e00' <= c <= u'\u9fa5':
            chinese_char += 1
        else:
            alphabetic_char += 1

    L = 2 * chinese_char + alphabetic_char
    return L

def format_in_fixed_length(obj, L):
    t = str(obj)
    n = max(L - str_len(t), 0)
    f = ' ' * n + t
    return f

def show_distribution(data, sort_f=lambda x:len(x[1]), indent=2, num=6):
    if type(data) is list:
        cnts = collections.defaultdict(list)
        for i, d in enumerate(data):
            cnts[d].append(i)
    elif type(data) is collections.defaultdict:
        cnts = data
    else:
        cnts = {}

    if cnts:
        L = max(4, max(map(str_len, cnts.keys())))
        print(' '*indent + format_in_fixed_length("kind", L) + ' num  id')
        for k, v in sorted(cnts.items(), key=sort_f, reverse=True):
            print(' '*indent + f'{format_in_fixed_length(k, L)} {len(v):<4d} {", ".join(map(str, v[:num]))}')

def train_valid_test_split(x, y, train_ratio=0.8, valid_ratio=0.1):
    n = len(x)
    idxs = list(range(n))
    random.shuffle(idxs)
    train_idxs = idxs[:int(n*train_ratio)]
    valid_idxs = idxs[int(n*train_ratio):int(n*valid_ratio)]
    test_idxs = idxs[int(n*valid_ratio):]

    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    for i in train_idxs:
        train_x.append(x[i])
        train_y.append(y[i])
    for i in valid_idxs:
        valid_x.append(x[i])
        valid_y.append(y[i])
    for i in test_idxs:
        test_x.append(x[i])
        test_y.append(y[i])
    return train_x, train_y, valid_x, valid_y, test_x, test_y
