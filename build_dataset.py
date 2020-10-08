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

rules = [
        'spouse','uncle','aunt','nephew',
        'niece','brother','sister','grandfather',
        'grandmother','ancestor','cousin_sibling',
        'great_grandfather','great_grandmother','predecessor'
        'descendant'
        ]

MAX_PADDING = 4

all_triples = []
all_traces = []

for rule in rules:

    triples,traces = utils.parse_ttl(
        file_name=os.path.join('.','data','traces',rule+'.ttl'),
        max_padding=MAX_PADDING)

    all_triples.append(triples)
    all_traces.append(traces)

all_triples = np.array(all_triples)
print(f"all_triples shape: {all_triples.shape}")

all_traces = np.array(all_traces)
print(f"all_traces shape: {all_traces.shape}")

exp_entities = np.array([[all_traces[:,i,:][:,0],
    all_traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

exp_relations = np.array([all_traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

entities = np.unique(np.concatenate([all_triples[:,0], all_triples[:,2], exp_entities],axis=0))
relations = np.unique(np.concatenate([all_triples[:,1], exp_relations],axis=0))

np.savez(os.path.join('.','data','royalty.npz'),
    all_triples=all_triples,all_traces=all_traces,
    entities=entities,relations=relations)

print('Dataset built.')

