#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random as rn
import os
import utils
import joblib
import rdflib
from sklearn.model_selection import train_test_split

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

g = rdflib.Graph()
g.parse("/Users/nhalliwe/Downloads/sparql",format="xml")

traces = utils.parse_traces('/Users/nhalliwe/Desktop/traces/spouse.ttl')

full_royalty = []

for subj,pred,obj in g:
    
    s = subj.split('/')[-1]
    p = pred.split('/')[-1]
    o = obj.split('/')[-1]
    
    if 'with_no_name_entry' in s or 'with_no_name_entry' in o:
        continue
        
    full_royalty.append((s,p,o))

royalty = []
explanations = []

for k in full_royalty:
    if traces[k]:
        royalty.append(k)
        explanations.append(traces[k])

royalty = np.array(royalty)
explanations = np.array(explanations).reshape(-1,3)

print(f"number of total observations {len(full_royalty)}")
print(f"number of observations with explanations {len(royalty)}")

X_train, X_test, exp_train, exp_test = train_test_split(royalty, explanations, test_size=0.33, random_state=42)

train = np.concatenate([X_train,exp_train], axis=0)

np.savez('/Users/nhalliwe/Desktop/Explain-KG/data/royalty.npz',
    X_train=X_train, X_test=X_test, exp_train=exp_train, exp_test=exp_test,train=train)