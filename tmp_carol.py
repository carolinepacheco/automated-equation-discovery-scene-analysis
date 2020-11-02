#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:53:51 2020

@author: carolinepacheco
"""

import os
import sys
import cv2
import glob
import time
import math
import argparse

#%%
import numpy as np
import pandas as pd
#import pybgs as bgs
import logging as log


from numba import jit

import chocolate as choco

if sys.version_info >= (3, 0):
    from six.moves import xrange
    
#%% process_structure

eq_symbols = [chr(i) for i in range(32,127)] # 95
ss_symbols = ['+', '-', '*', '/'] # 4
no_symbols = ['(', 'Z', 'C', 'a', ')', ' ', '"', "'", ',', '.', '\\', '[', ']', '`', '^', '{', '}', ':', ';', '-', '=', '<', '>', '~', '|', '_'] + ss_symbols # 30
va_symbols = list(set(eq_symbols) - set(no_symbols)) # 66 (max 33)
va_symbols.sort()

filepath = 'dataset/data.txt'

if not os.path.isfile(filepath):
    print("File " + str(filepath) + " does not exists!")
    
    
with open(filepath) as fp:
    for i, equation in enumerate(fp):
        # print("Line {}: {}".format(i, equation))
        equation = equation.replace("\n", "")
        
        
eq_split = equation.split("o") # ['((Z', 'C)', '(Z', 'C))']
search_space = {}
new_equation = eq_split[0]
for j in range(len(eq_split)-1):
        # j_symbol = eq_symbols[j]
        j_symbol = va_symbols[j]
        search_space[j_symbol] = choco.choice(ss_symbols)
        new_equation = new_equation + j_symbol + eq_split[j+1]
        
# linear:
# max_iterations = (len(eq_split)-1) * len(ss_symbols)
#
# exponential:
max_iterations = int(math.pow(len(ss_symbols), (len(eq_split)-1)))  

if max_iterations > 1024:
        max_iterations = 1024      
        

#%% mutate_equation
import random

i = random.randrange(100,200)
database_url = "sqlite:///db/chocolate_"+str(i)+".db"
conn = choco.SQLiteConnection(url=database_url)
conn.clear()
sampler = choco.MOCMAES(conn, search_space, mu=2)
best_loss = 1
best_params = None
best_equation = None
for n in range(max_iterations):
    token, params = sampler.next()
    print(new_equation + " iter " + str(n+1) + " of " + str(max_iterations))
    
    
#%% score_equation


if params is not None:
    for key in params:
        #print(key, params[key])
        equation = new_equation.replace(key, params[key])
        print(" scoring mutation: " + equation)


#%%
import pybgs as bgs
img_in_folder = 'dataset/skating/input'
img_in_array = sorted(glob.iglob(img_in_folder + '/*.jpg'))
print(" in folder " + img_in_folder + " " + str(len(img_in_array)))
img_gt_folder = 'dataset/skating/groundtruth'
img_gt_array = sorted(glob.iglob(img_gt_folder + '/*.png'))
print("gt folder " + img_gt_folder + " " + str(len(img_gt_array)))

algorithm = bgs.LBP_MRF()
    