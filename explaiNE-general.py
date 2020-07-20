#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pandas as pd
import random as rn
import os
import cne
import maxent
from scipy import sparse
from scipy.stats import halfnorm

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

triples = [('Eve', 'type', 'Lecturer'),
           #('Eve', 'type', 'Person'), 
           ('Lecturer', 'subClassOf', 'Person'), 
           #('David', 'type', 'Person'),
           ('David', 'type', 'Researcher'),
           ('Researcher', 'subClassOf', 'Person'),
           ('Flora', 'hasSpouse', 'Gaston'),
           ('Gaston', 'type', 'Person'),
           #('Flora', 'type', 'Person')
          ]

train = np.array(triples)

entities = np.unique(np.concatenate((train[:,0], train[:,2]), axis=0)).tolist()
relations = np.unique(train[:,1]).tolist()

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

idx2ent = {idx:ent for ent,idx in ent2idx.items()}
idx2rel = {idx:rel for rel,idx in rel2idx.items()}

embedding_dim = 10

A = np.zeros(shape=(num_entities,num_entities))

for h,r,t in train:
    
    h_idx = entities.index(h)
    r_idx = relations.index(r)
    t_idx = entities.index(t)
    
    A[h_idx,r_idx ,t_idx] = 1