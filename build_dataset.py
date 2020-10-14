#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import utils
import argparse

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

# rules = [
#         'spouse','uncle','aunt','nephew',
#         'niece','brother','sister','grandfather',
#         'grandmother','great_ancestor','cousin_sibling',
#         'great_grandfather','great_grandmother','predecessor'
#         'descendant'
#         ]

rules = ['grandmother','grandfather']
MAX_PADDING = 3

data = dict()
all_triples = []
all_traces = []

for rule in rules:

    triples,traces = utils.parse_ttl(
        file_name=os.path.join('.','data','traces',rule+'.ttl'),
        max_padding=MAX_PADDING)
    
    all_triples.append(triples)
    all_traces.append(traces)
    
    data[rule + '_triples'] = triples
    data[rule + '_traces'] = traces

all_triples = np.concatenate(all_triples,axis=0)
print(f"all_triples shape: {all_triples.shape}")

all_traces = np.concatenate(all_traces,axis=0)
print(f"all_traces shape: {all_traces.shape}")

exp_entities = np.array([[all_traces[:,i,:][:,0],
    all_traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

exp_relations = np.array([all_traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

all_entities = np.unique(np.concatenate([all_triples[:,0], all_triples[:,2], exp_entities],axis=0))
all_relations = np.unique(np.concatenate([all_triples[:,1], exp_relations],axis=0))

data['entities'] = all_entities
data['relations'] = all_relations

np.savez(os.path.join('.','data','royalty.npz'),**data)

print('Dataset built.')

