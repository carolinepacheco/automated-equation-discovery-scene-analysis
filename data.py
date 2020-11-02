import torch

filename = 'dataset/equations.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
lines = raw_text.split('\n')
train_data = [item for item in lines if item != '']

# ' <$>' to indicate the beginning of equation
# '<#>' to indicate the end of an equation
# '<pad>' to make all equations of the same length 

chars = set()
for string in train_data:
    chars.update(string)
all_sys = sorted(list(chars)) + ['<$>', '<#>', '<pad>']

vocab = all_sys
c2i = {c: i for i, c in enumerate(all_sys)}
i2c = {i: c for i, c in enumerate(all_sys)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vector = torch.eye(len(c2i))

def char2id(char):
    return c2i[char]


def id2char(id):
    if id not in i2c:
        return i2c[32]
    else:
        return i2c[id]


def string2ids(string, add_bos=False, add_eos=False):
    ids = [char2id(c) for c in string]
    if add_bos:
        ids = [c2i['<$>']] + ids
    if add_eos:
        ids = ids + [c2i['<#>']]
    return ids


def ids2string(ids, rem_bos=True, rem_eos=True):
    if len(ids) == 0:
        return ''
    if rem_bos and ids[0] == c2i['<$>']:
        ids = ids[1:]
    if rem_eos and ids[-1] == c2i['<#>']:
        ids = ids[:-1]
    string = ''.join([id2char(id) for id in ids])
    return string


def string2tensor(string, device='model'):
    ids = string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long, device=device if device == 'model' else device)
    return tensor


tensor = [string2tensor(string, device=device) for string in train_data]
vector = torch.eye(len(c2i))
