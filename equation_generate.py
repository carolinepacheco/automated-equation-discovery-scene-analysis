"""
Created on Sat Sep 26 12:47:17 2020

@author: Caroline Pacheco
@email: lolyne.pacheco@gmail.com
"""

import pandas as pd
from vae_model import VAE
from trainer import *
from samples import *
from data import *
from LBP.eq_validity import equation_validity
import collections
import random

# train a vae instance model
model = VAE(vocab, vector).to(device)
fit(model, train_data)

# generate sample equations
sample = sample.take_samples(model, n_batch, 3000)
# validate generated equations
valid, not_valid, max_equations, unseenval = equation_validity(sample)

# save output file
df = pd.DataFrame(unseenval)
df.to_csv(r'output/my_equation.txt', header=None, index=None, sep='\t', mode='a')
