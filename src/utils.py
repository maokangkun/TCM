import os
import torch
import random
import collections
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from rich.console import Console
from rich.traceback import install
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

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
    valid_idxs = idxs[int(n*train_ratio):int(n*(train_ratio+valid_ratio))]
    test_idxs = idxs[int(n*(train_ratio+valid_ratio)):]

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

def test_example(tokenizer, mlm_model, feat, index=0):
    example = feat[index]
    print(f'example: {example}')
    tokenized_text = tokenizer.tokenize(example)
    token_idx = tokenizer.convert_tokens_to_ids(tokenized_text)
    token_tensor = torch.tensor([token_idx])

    encode_idx = tokenizer.encode(example)
    encode_text = ''.join(tokenizer.convert_ids_to_tokens(encode_idx))
    encode_tensor = torch.tensor([encode_idx])

    encode_out = tokenizer.batch_encode_plus(feat, padding=True)
    encode_idx2 = encode_out['input_ids'][index]
    encode_text2 = ''.join(tokenizer.convert_ids_to_tokens(encode_idx2))
    encode_tensor2 = torch.tensor([encode_idx2])
    token_type_ids_tensor = torch.tensor([encode_out['token_type_ids'][index]])
    attention_mask_tensor = torch.tensor([encode_out['attention_mask'][index]])

    out = mlm_model(token_tensor)
    out_token_idx = out.logits[0].argmax(1).tolist()
    out_token = tokenizer.convert_ids_to_tokens(out_token_idx)
    for i in range(len(tokenized_text)):
        if out_token[i] != tokenized_text[i]:
            out_token[i] = '[red]' + out_token[i] + '[/]'
    out_text = ''.join(out_token)

    out2 = mlm_model(encode_tensor)
    out2_token_idx = out2.logits[0].argmax(1).tolist()
    out2_token = tokenizer.convert_ids_to_tokens(out2_token_idx)
    for i in range(len(tokenized_text)):
        if out2_token[i+1] != tokenized_text[i]:
            out2_token[i+1] = '[red]' + out2_token[i+1] + '[/]'
    out2_text = ''.join(out2_token)

    out3 = mlm_model(encode_tensor2, token_type_ids=token_type_ids_tensor, attention_mask=attention_mask_tensor)
    out3_token_idx = out3.logits[0].argmax(1).tolist()
    out3_token = tokenizer.convert_ids_to_tokens(out3_token_idx)
    for i in range(len(tokenized_text)):
        if out3_token[i+1] != tokenized_text[i]:
            out3_token[i+1] = '[red]' + out3_token[i+1] + '[/]'
    out3_text = ''.join(out3_token)

    print(f'tokenized: {tokenized_text}')
    print(f'tokenized index: {token_idx}')
    print(f'encode index: {encode_idx}')
    print(f'encode text: {encode_text}')
    print(f'batch encode index ({len(encode_idx2)}): {encode_idx2}')
    print(f'batch encode text ({len(encode_idx2)}): {encode_text2}')
    print(f'model out: {out_token_idx}')
    print(f'model out text: {out_text}')
    print(f'model out (encode): {out2_token_idx}')
    print(f'model out text: {out2_text}')
    print(f'model out (batch): {out2_token_idx}')
    print(f'model out text: {out2_text}')

class TCMDataset(Dataset):
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        feat = self.feat[idx]
        label = self.label[idx]
        return feat, label

def plot(result, out):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for k in result:
        ax1.plot(result[k][0], label=k)
        ax1.set_title('Loss')
        ax1.legend()

        ax2.plot(result[k][1], label=k)
        ax2.set_title('Accuracy')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f'log/{out}.png', dpi=300)
