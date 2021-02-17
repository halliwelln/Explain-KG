#!/usr/bin/env python3

import numpy as np
import utils
import os
import tensorflow as tf
import joblib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('rule',type=str,help=
    'Enter which rule to use spouse,successor,...etc (str), full_data for full dataset')
parser.add_argument('trace_length',type=int)
args = parser.parse_args()

RULE = args.rule
TRACE_LENGTH = args.trace_length

data = np.load(os.path.join('..','data','royalty.npz'))

triples,traces,nopred,entities,relations = utils.get_data(data,RULE)

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

idx2ent = dict(zip(range(NUM_ENTITIES),entities))
idx2rel = dict(zip(range(NUM_RELATIONS),relations))

explaine_data = np.load(
    os.path.join('..','data','preds','explaine_'+RULE+'_preds.npz'),
    allow_pickle=True
)

true_triples = triples[explaine_data['test_idx']]
true_traces = traces[explaine_data['test_idx']][:,0:TRACE_LENGTH,:]

pred_traces = utils.idx2array(explaine_data['preds'],idx2ent,idx2rel)

adj_data = np.concatenate([triples,traces[:,0:TRACE_LENGTH,:].reshape(-1,3)],axis=0)

adj_data_sparse = utils.array2idx(adj_data,ent2idx,rel2idx)

adj_mats = utils.get_adj_mats(
    data=adj_data_sparse,
    num_entities=NUM_ENTITIES,
    num_relations=NUM_RELATIONS
)

def get_count(i,true_triples,pred_traces,ent2idx,adj_mats,num_relations):
    
    current_count = 0
    
    head,_,tail = true_triples[i]
    pred_i = pred_traces[i]
    
    head_idx = ent2idx[head]
    tail_idx = ent2idx[tail]
    
    neighbor_indices = []
    
    for rel_idx in range(num_relations):
        
        dense_mat = tf.sparse.to_dense(adj_mats[rel_idx]).numpy()[0]
        
        head_neighbors = np.argwhere(dense_mat[head_idx,:]).flatten()
        tail_neighbors = np.argwhere(dense_mat[:,tail_idx]).flatten()
        
        neighbor_indices += head_neighbors.tolist()
        neighbor_indices += tail_neighbors.tolist()
    
    neighbors = [idx2ent[idx] for idx in neighbor_indices]
    
    pred_entities = np.unique(np.concatenate((pred_i[:,0],pred_i[:,2]),axis=0)).tolist()
    
    for p in pred_entities:
        if p in neighbors:
            current_count += 1
            break
    
    if current_count >= 1:
        return 1
    else:
        return 0

unique = []

for list_ in pred_traces[:,:,1]:
    list_ = list(list_)
    if list_ not in unique:
        
        tup = tuple(list_)
        
        unique.append(tup)
        
d = {}

for tup in unique:
        
    count = (tup == pred_traces[:,:,1]).all(axis=1).sum() 
    
    d[tup] = count
    
output = sorted(d.items(),key=lambda x:x[1],reverse=True)

print(f"3 most predicted relation pairs for {RULE}: {output[0:3]}")

total_count = joblib.Parallel(n_jobs=-2, verbose=20)(
            joblib.delayed(get_count)(i,true_triples,pred_traces,ent2idx,adj_mats,num_relations=NUM_RELATIONS)
                for i in range(len(true_triples))
            )

num_triples = true_triples.shape[0]

print(f"Percentage of triples using 1st degree neighbors of head or tail: {sum(total_count) / num_triples}")

# count = 0
# for tup in output:
#     if 'brother' not in tup[0] or 'parent' not in tup[0]:
#         count += tup[1]

# print(f"Percentage of triples that did not include brother or parent {count/num_triples} ")

####CHECK LENGTH 1 TRACES
