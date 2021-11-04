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

parser = argparse.ArgumentParser()
parser.add_argument("list", nargs="+", 
    help='grandparent spouse, or spouse successor predecessor')
args = parser.parse_args()

MAX_PADDING = 2

rules = args.list

data = dict()
all_triples = []
all_traces = []

for rule in rules:

    if (rule == 'brother') or (rule == 'sister'):
        rule_file = 'brother_sister'

    elif (rule == 'successor') or (rule == 'predecessor'):
        rule_file = 'successor_predecessor'

    else:
        rule_file = rule

    triples,traces = utils.parse_ttl(
        file_name=os.path.join('..','data','traces',rule_file+'.ttl'),
        max_padding=MAX_PADDING
    )

    _, unique_idx = np.unique(triples, axis=0,return_index=True)

    triples = triples[unique_idx]
    traces = traces[unique_idx]
    
    #get indicies of triples for <rule>
    idx = triples[:,1] == rule

    triples = triples[idx]
    traces = traces[idx]

    #replace male/female triples with unknown
    if rule_file == 'brother_sister':
        num_triples = traces[:,2,:].shape[0]
        traces[:,2,:] = np.array([['UNK_ENT', 'UNK_REL', 'UNK_ENT'] for i in range(num_triples)])

    exp_entities = np.array([[traces[:,i,:][:,0],
        traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

    exp_relations = np.array([traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

    if rule == 'spouse':

        swapped_triples = triples[:,[2,1,0]]
        swapped_traces = traces[:,:,[2,1,0]]

        triples = np.concatenate([triples,swapped_triples],axis=0)
        traces = np.concatenate([traces,swapped_traces],axis=0)
    
    all_triples.append(triples)
    all_traces.append(traces)

    data[rule + '_triples'] = triples
    data[rule + '_traces'] = traces
    data[rule + '_entities'] = np.unique(np.concatenate([triples[:,0], triples[:,2], exp_entities],axis=0))
    data[rule + '_relations'] = np.unique(np.concatenate([triples[:,1], exp_relations],axis=0))

all_triples = np.concatenate(all_triples,axis=0)
print(f"all_triples shape: {all_triples.shape}")

all_traces = np.concatenate(all_traces,axis=0)
print(f"all_traces shape: {all_traces.shape}")

all_exp_entities = np.array([[all_traces[:,i,:][:,0],
    all_traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

all_exp_relations = np.array([all_traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

all_entities = np.unique(np.concatenate([all_triples[:,0], all_triples[:,2], all_exp_entities],axis=0))
all_relations = np.unique(np.concatenate([all_triples[:,1], all_exp_relations],axis=0))

data['all_entities'] = all_entities
data['all_relations'] = all_relations
data['rules'] = rules

print('Saving numpy file...')

if rules == ['grandparent', 'spouse']:
    number = str(30)
elif rules == ['spouse', 'successor', 'predecessor']:
    number = str(20)

np.savez(os.path.join('..','data',f'royalty_{number}k.npz'),**data)

print('Done')

# ttl_dir = os.path.join('..','data','traces')
# ttl_files = [f for f in os.listdir(ttl_dir) if f.endswith('.ttl')]
# print(ttl_dir)
# print(ttl_files)
# g = Graph()

# for file_name in ttl_files:
#     print(os.path.join(ttl_dir,file_name))
#     g.parse(os.path.join(ttl_dir,file_name),format='nt')

# print(len(g))
#print('Saving rdf file...')
# #g.serialize(destination=os.path.join('..','data','rules','royalty'),format='xml')