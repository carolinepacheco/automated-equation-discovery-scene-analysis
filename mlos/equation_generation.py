__file__ = variables.get("PA_TASK_NAME")

import sys
import json
import wget
import uuid
from os.path import basename, splitext, exists, join

if sys.version_info[0] >= 3:
    unicode = str

######################## AUTOML SETTINGS ##########################
# MOCMAES SAERCH SPACE
# [{"ENC_HIDDEN": choice([125, 256, 512]),
#   "DEC_HIDDEN": choice([512, 800]),
#   "ENC_LAYERS": choice([1, 2, 4, 6]),
#   "DEC_LAYERS": choice([1, 2, 4, 6]),
#   "ENC_DROPOUT": choice([0.01, 0.02, 0.01, 0.1, 0.2]),
#   "DEC_DROPOUT": choice([0.01, 0.02, 0.01, 0.1, 0.2]),
#   "N_BATCH": choice([32, 64, 512]),
#   "LEARNING_RATE": choice([0.001, 0.005]),
#   "OPTIMIZER": choice(['optim.Adam', 'optim.Adadelta', 'optim.RMSpro'])}
# ]

DATA_PATH = str(variables.get("DATA_PATH"))
enc_hidden = int(variables.get("ENC_HIDDEN"))
dec_hidden = int(variables.get("DEC_HIDDEN"))
enc_layers = int(variables.get("ENC_LAYERS"))
dec_layers = int(variables.get("DEC_LAYERS"))
enc_dropout = float(variables.get("ENC_DROPOUT"))
dec_dropout = float(variables.get("DEC_DROPOUT"))
n_batch = int(variables.get("N_BATCH"))
optimizer_lr = variables.get("OPTIMIZER")
learning_rate = float(variables.get("LEARNING_RATE"))

input_variables = variables.get("INPUT_VARIABLES")
if input_variables is not None and input_variables != '':
    input_variables = json.loads(input_variables)
    enc_hidden = input_variables["ENC_HIDDEN"]
    dec_hidden = input_variables["DEC_HIDDEN"]
    enc_layers = input_variables["ENC_LAYERS"]
    dec_layers = input_variables["DEC_LAYERS"]
    enc_dropout = input_variables["ENC_DROPOUT"]
    dec_dropout = input_variables["DEC_DROPOUT"]
    n_batch = input_variables["N_BATCH"]
    optimizer_lr = input_variables["OPTIMIZER"]
    learning_rate = input_variables["LEARNING_RATE"]

# Get current job ID
PA_JOB_ID = variables.get("PA_JOB_ID")

# Check parent job ID
PARENT_JOB_ID = genericInformation.get('PARENT_JOB_ID')

# Define the path to save the model
OUTPUT_PATH = variables.get("DOCKER_LOG_PATH")
MODEL_PATH = join(OUTPUT_PATH, 'model')
os.makedirs(MODEL_PATH, exist_ok=True)

### BEGIN VISDOM ###
VISDOM_ENABLED = variables.get("VISDOM_ENABLED")
if VISDOM_ENABLED is not None and VISDOM_ENABLED.lower() == "true":
    from visdom import Visdom

VISDOM_ENDPOINT = variables.get("VISDOM_ENDPOINT")
if VISDOM_ENDPOINT is not None:
    from visdom import Visdom

    VISDOM_ENDPOINT = VISDOM_ENDPOINT.replace("http://", "")
    print("VISDOM_ENDPOINT: ", VISDOM_ENDPOINT)
    (VISDOM_HOST, VISDOM_PORT) = VISDOM_ENDPOINT.split(":")

    print("VISDOM_HOST: ", VISDOM_HOST)
    print("VISDOM_PORT: ", VISDOM_PORT)

    print("Connecting to %s:%s" % (VISDOM_HOST, VISDOM_PORT))
    vis = Visdom(server="http://" + VISDOM_HOST, port=int(VISDOM_PORT))
    assert vis.check_connection()

env = 'main'
if PARENT_JOB_ID is not None:
    env = 'job_id_' + PARENT_JOB_ID
###################################################################

print("DATA_PATH: " + DATA_PATH)

if DATA_PATH is not None and DATA_PATH.startswith("http"):
    # Get an unique ID
    ID = str(uuid.uuid4())

    # Define localspace
    LOCALSPACE = join('data', ID)
    os.makedirs(LOCALSPACE, exist_ok=True)
    print("LOCALSPACE:  " + LOCALSPACE)

    DATASET_NAME = splitext(DATA_PATH[DATA_PATH.rfind("/") + 1:])[0]
    DATASET_PATH = join(LOCALSPACE, DATASET_NAME)
    os.makedirs(DATASET_PATH, exist_ok=True)

    print("Dataset information: ")
    print("DATASET_NAME: " + DATASET_NAME)
    print("DATASET_PATH: " + DATASET_PATH)

    print("Downloading...")
    filename = wget.download(DATA_PATH, DATASET_PATH)
    print("FILENAME: " + filename)
    print("OK")

#############################################################################
# data file
import torch

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

#############################################################################
# vae_model file
import torch
import torch.nn as nn
import torch.nn.functional as F

# vae parameters
q_bidir = True
d_z = 128


class VAE(nn.Module):
    def __init__(self, vocab, vector):
        super().__init__()
        self.vocabulary = vocab
        self.vector = vector

        n_vocab, d_emb = len(vocab), vector.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, c2i['<pad>'])
        self.x_emb.weight.data.copy_(vector)

        # Encoder
        self.encoder_rnn = nn.GRU(d_emb, enc_hidden, num_layers=enc_layers, batch_first=True,
                                  dropout=enc_dropout if enc_layers > 1 else 0, bidirectional=q_bidir)
        q_d_last = enc_hidden * (2 if q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, d_z)
        self.q_logvar = nn.Linear(q_d_last, d_z)

        # Decoder
        self.decoder_rnn = nn.GRU(d_emb + d_z, dec_hidden, num_layers=dec_layers, batch_first=True,
                                  dropout=dec_dropout if dec_layers > 1 else 0)
        self.decoder_latent = nn.Linear(d_z, dec_hidden)
        self.decoder_fullyc = nn.Linear(dec_hidden, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_rnn, self.q_mu, self.q_logvar])
        self.decoder = nn.ModuleList([self.decoder_rnn, self.decoder_latent, self.decoder_fullyc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device if device == 'model' else device)
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def forward(self, x):
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        # print("forward")
        return kl_loss, recon_loss

    def forward_encoder(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def forward_decoder(self, x, z):
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=c2i['<pad>'])
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_latent(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fullyc(output)

        recon_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)), x[:, 1:].contiguous().view(-1),
                                     ignore_index=c2i['<pad>'])
        return recon_loss

    def sample_z_prior(self, n_batch):
        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
                z = z.to(self.device)
                z_0 = z.unsqueeze(1)
                h = self.decoder_latent(z)
                h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
                w = torch.tensor(c2i['<$>'], device=self.device).repeat(n_batch)
                x = torch.tensor([c2i['<pad>']], device=device).repeat(n_batch, max_len)
                x[:, 0] = c2i['<$>']
                end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
                eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)
                # eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

                for i in range(1, max_len):
                    x_emb = self.x_emb(w).unsqueeze(1)
                    x_input = torch.cat([x_emb, z_0], dim=-1)

                    o, h = self.decoder_rnn(x_input, h)
                    y = self.decoder_fullyc(o.squeeze(1))
                    y = F.softmax(y / temp, dim=-1)

                    w = torch.multinomial(y, 1)[:, 0]
                    x[~eos_mask, i] = w[~eos_mask]
                    i_eos_mask = ~eos_mask & (w == c2i['<#>'])
                    end_pads[i_eos_mask] = i + 1
                    eos_mask = eos_mask | i_eos_mask

                    new_x = []
                    for i in range(x.size(0)):
                        new_x.append(x[i, :end_pads[i]])

        return [self.tensor2string(i_x) for i_x in new_x]


#############################################################################
# trainer file

from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import math
import ast
from collections import UserList, defaultdict
from math import *

# training parameters
n_last = 4000
kl_start = 0.05
kl_w_start = 0.0
kl_w_end = 1.0
n_epoch = 150
n_workers = 0

clip_grad = 100
lr_start = 0.005
lr_n_period = 40
lr_n_mult = 1
lr_end = 3 * 1e-4
lr_n_restarts = 10


def _n_epoch():
    return sum(lr_n_period * (lr_n_mult ** i) for i in range(lr_n_restarts))


def _train_epoch(model, epoch, train_loader, kl_weight, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    kl_loss_values = CircularBuffer(n_last)
    recon_loss_values = CircularBuffer(n_last)
    loss_values = CircularBuffer(n_last)
    for i, input_batch in enumerate(train_loader):
        input_batch = tuple(data.to(device) for data in input_batch)

        # forward
        kl_loss, recon_loss = model(input_batch)
        loss = kl_weight * kl_loss + recon_loss
        # backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(get_optim_params(model), clip_grad)
            optimizer.step()

        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        lr = (optimizer.param_groups[0]['lr'] if optimizer is not None else None)

        # update train_loader
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'loss={loss_value:.5f}', f'(kl={kl_loss_value:.5f}', f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}']
    postfix = {'epoch': epoch, 'kl_weight': kl_weight, 'lr': lr, 'kl_loss': kl_loss_value,
               'recon_loss': recon_loss_value, 'loss': loss_value, 'mode': 'Eval' if optimizer is None else 'Train'}
    return postfix


def _train(model, train_loader, val_loader=None, logger=None):
    param = str(optimizer_lr) + "(get_optim_params(model))"
    optimizer = eval(param)
    optimizer.param_groups[0]['lr'] = learning_rate

    lr_annealer = CosineAnnealingLRWithRestart(optimizer)

    model.zero_grad()

    n_epochs_stop = 60
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.Inf

    for epoch in range(n_epoch):
        iter = 0
        val_loss = 0
        kl_annealer = KLAnnealer(n_epoch)
        kl_weight = kl_annealer(epoch)
        postfix = _train_epoch(model, epoch, train_loader, kl_weight, optimizer)
        print("%d[Loss: %.5f Kl-loss: %.5f Recon-loss: %.5f]" % (
            postfix['epoch'], postfix['loss'], postfix['kl_loss'], postfix['recon_loss']))
        lr_annealer.step()

        # Update parameters
        val_loss += postfix['loss']
        val_loss = val_loss / len(train_loader)

        # early stopping
        if val_loss < min_val_loss:
            file_path = join(MODEL_PATH, 'weights-' + str(PA_JOB_ID) + '.hdf5')
            torch.save(model.state_dict(), file_path)
            epochs_no_improve = 0
            min_val_loss = val_loss

        else:
            epochs_no_improve += 1

        iter += 1
        if epoch > 59 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue
        break

        if early_stop:
            print("Stopped")
            break


def fit(model, train_data, val_data=None):
    logger = Logger() if False is not None else None
    train_loader = get_dataloader(model, train_data, shuffle=True)

    val_loader = None if val_data is None else get_dataloader(model, val_data, shuffle=False)
    _train(model, train_loader, val_loader, logger)
    return model


def get_collate_device(model):
    return model.device


def get_dataloader(model, train_data, collate_fn=None, shuffle=True):
    if collate_fn is None:
        collate_fn = get_collate_fn(model)
        print(collate_fn)
    return DataLoader(train_data, batch_size=n_batch, shuffle=shuffle, num_workers=n_workers, collate_fn=collate_fn)


def get_collate_fn(model):
    device = get_collate_device(model)

    def collate(train_data):
        train_data.sort(key=len, reverse=True)
        tensors = [string2tensor(string, device=device) for string in train_data]
        return tensors

    return collate


def get_optim_params(model):
    return (p for p in model.parameters() if p.requires_grad)


class KLAnnealer:
    def __init__(self, n_epoch):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer):
        self.n_period = lr_n_period
        self.n_mult = lr_n_mult
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        return self.data.mean()


class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return Logger(self.data[key])
        else:
            ldata = self.sdata[key]
            if isinstance(ldata[0], dict):
                return Logger(ldata)
            else:
                return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)


#############################################################################
# sample file

import pandas as pd
from tqdm import tqdm

n_jobs = 1
max_len = 100


class sample():
    def take_samples(model, n_batch, n_samples):
        n = n_samples
        samples = []
        with tqdm(total=n_samples, desc='Generating equations') as T:
            while n > 0:
                current_samples = model.sample(min(n, n_batch), max_len)
                samples.extend(current_samples)
                n -= len(current_samples)
                T.update(len(current_samples))
        # samples = pd.DataFrame(samples, columns=['SMILES'])
        samples = pd.DataFrame(samples, columns=['EQUATIONS'])
        return samples


#############################################################################
# computed lbp file

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def compute_LBP(imgnm, lbp_str):
    image = plt.imread(imgnm)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3
    s = 0
    for ih in range(0, image.shape[0] - neighboor):  # loop by line
        for iw in range(0, image.shape[1] - neighboor):
            ### Get natrix image 3 by 3 pixel
            a = 0.01
            Z = gray_image[ih:ih + neighboor, iw:iw + neighboor]
            C = Z[1, 1]
            s = eval(lbp_str)

            lbp = (s >= 0) * 1.0

            ### Convert the binary operated values to a decimal (digit)
            where_lbp_vector = np.where(lbp)[0]

            num = np.sum(2 ** where_lbp_vector) if len(where_lbp_vector) >= 1 else 0
            imgLBP[ih + 1, iw + 1] = num

    return image, imgLBP


#############################################################################
# eq_validaty file

import re
import random


def valid_eq(eq):
    Z = random.randint(1, 9)
    C = random.randint(1, 9)
    a = random.randint(1, 9)
    try:
        exec (eq)
        return True
    except ZeroDivisionError:
        return False
    except SyntaxError:
        return False
    except TypeError:
        return False
    except NameError:
        return False
    except MemoryError:
        return False


def valid_expression(eq):
    eq = eq.replace("o", "+")
    matched = re.match(
        '^(\()(\()?(.+)(\))?(\))(.*)|^(\()(\()(.+)(\))(\))(.*)$', eq)
    if (bool(matched)):
        return True
    else:
        return False


def equation_validity(equation):
    print("----- BEGIN equation_validity -----")
    valid = []
    not_valid = []

    for i, eq2 in enumerate(equation['EQUATIONS']):
        eq = eq2.replace("o", "+")
        val_exp = valid_expression(eq)
        val_exec = valid_eq(eq)
        if (val_exp and val_exec):
            print('Valid: ', eq2)
            valid.append(eq2)
        else:
            val_exec = valid_eq(eq)
            if (val_exec):
                print('Valid: ', eq2)
                valid.append(eq2)
            else:
                print('Not valid: ', eq2)
                not_valid.append(eq2)

    max_equations = len(valid) + len(not_valid)
    # unique valid equations
    unique_equations = list(set(valid))

    raw_split = re.split("#", raw_text)
    check_end = [i.replace("$", "") for i in raw_split]
    find_equation = [item for item in check_end if item != '']

    # unique unseen valid equations
    N = [i for i in range(len(unique_equations)) if not unique_equations[i] in find_equation]
    unseenval = []
    for i in N:
        unseenval.append(unique_equations[i])

    print('Genrated equations', max_equations)
    print('Valid equations', len(valid))
    print('Unique valid equations', len(unique_equations))
    print('Unseen valid equations', len(N))

    return valid, not_valid, max_equations, unseenval


def take_results(valid, not_valid):
    max_equations = len(valid) + len(not_valid)
    unique_equations = list(set(valid))

    raw_split = re.split("#", raw_text)
    check_end = [i.replace("$", "") for i in raw_split]
    find_equation = [item for item in check_end if item != '']

    N = [i for i in range(len(unique_equations)) if not unique_equations[i] in find_equation]

    print('Genrated equations', max_equations)
    print('Valid equations', len(valid))
    print('Unique valid equations', len(set(valid)))
    print('Unseen valid equations', len(N))


#############################################################################
# CALL FUNCTIONS
#############################################################################
import collections
import random

# create a vae model
model = VAE(vocab, vector).to(device)
fit(model, train_data)

# generate sample equations
sample = sample.take_samples(model, n_batch, 300)
# validate generated equations
valid, not_valid, max_equations, unseenval = equation_validity(sample)

max_equations = len(valid) + len(not_valid)

x = (len(unseenval) / max_equations)
loss = 1 - x

######################## AUTOML SETTINGS ##########################
# """
# To appear in Job Analytics
resultMap.put("LOSS", str(loss))
# resultMap.put("LOSS_VAE", str(g_loss))
resultMap.put("EQ_TOTAL", str(max_equations))
resultMap.put("EQ_VALID", str(len(valid)))
resultMap.put("EQ_UNSEEN_VALID", str(len(unseenval)))
# resultMap.put("NOTVALID", str(len(not_valid)))

token = variables.get("TOKEN")
# Convert from JSON to dict
token = json.loads(token)

# Return the loss value
result_map = {'token': token, 'loss': loss}
print('result_map: ', result_map)

resultMap.put("RESULT_JSON", json.dumps(result_map))
# """
###################################################################
