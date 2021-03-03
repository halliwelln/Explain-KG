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
parser.add_argument('method',type=str, help='gnn_explainer or explaine') 
args = parser.parse_args()

RULE = args.rule
TRACE_LENGTH = args.trace_length
METHOD = args.method

data = np.load(os.path.join('..','data','royalty.npz'))

triples,traces,nopred,entities,relations = utils.get_data(data,RULE)

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

idx2ent = dict(zip(range(NUM_ENTITIES),entities))
idx2rel = dict(zip(range(NUM_RELATIONS),relations))

pred_data = np.load(
    os.path.join('..','data','preds', METHOD+'_'+RULE+'_preds.npz'),
    allow_pickle=True
)

true_triples = triples[pred_data['test_idx']]
true_traces = traces[pred_data['test_idx']][:,0:TRACE_LENGTH,:]

if METHOD == 'explaine':
    pred_traces = utils.idx2array(pred_data['preds'],idx2ent,idx2rel)
elif METHOD == 'gnn_explainer':
    pred_traces = []
    for i in range(len(pred_data['preds'])):

        preds_i = []

        for rel_idx in range(NUM_RELATIONS):

            triples_i = pred_data['preds'][i][rel_idx]

            if triples_i.shape[0]:
                rel_indices = (np.ones((triples_i.shape[0],1)) * rel_idx).astype(np.int64)
                concat = np.concatenate([triples_i,rel_indices],axis=1)
                preds_i.append(concat[:,[0,2,1]])
        preds_i = np.concatenate(preds_i,axis=0)
        pred_traces.append(utils.idx2array(preds_i,idx2ent,idx2rel))

    pred_traces = np.array(pred_traces,dtype=object)

# adj_data = np.concatenate([triples,traces[:,0:TRACE_LENGTH,:].reshape(-1,3),nopred],axis=0)

# adj_data_sparse = utils.array2idx(adj_data,ent2idx,rel2idx)

# adj_mats = utils.get_adj_mats(
#     data=adj_data_sparse,
#     num_entities=NUM_ENTITIES,
#     num_relations=NUM_RELATIONS
# )

# def get_count(i,true_triples,pred_traces,ent2idx,adj_mats,num_relations):
    
#     current_count = 0
    
#     head,_,tail = true_triples[i]
#     pred_i = np.array(pred_traces[i])
    
#     head_idx = ent2idx[head]
#     tail_idx = ent2idx[tail]
    
#     neighbor_indices = []
    
#     for rel_idx in range(num_relations):
        
#         dense_mat = tf.sparse.to_dense(adj_mats[rel_idx]).numpy()[0]
        
#         head_neighbors = np.argwhere(dense_mat[head_idx,:]).flatten()
#         tail_neighbors = np.argwhere(dense_mat[:,tail_idx]).flatten()
        
#         neighbor_indices += head_neighbors.tolist()
#         neighbor_indices += tail_neighbors.tolist()
    
#     neighbors = [idx2ent[idx] for idx in neighbor_indices]
    
#     pred_entities = np.unique(np.concatenate((pred_i[:,0],pred_i[:,2]),axis=0)).tolist()
    
#     for p in pred_entities:
#         if p in neighbors:
#             current_count += 1
#             break
    
#     if current_count >= 1:
#         return 1
#     else:
#         return 0

true_triples = triples[pred_data['test_idx']]
true_traces = traces[pred_data['test_idx']][:,0:TRACE_LENGTH,:]

if RULE == 'spouse' and METHOD == 'explaine':
    pred_traces = utils.idx2array(pred_data['preds'],idx2ent,idx2rel)
    idx = np.argwhere(pred_traces[:,0,:][:,1] == 'sister')[0]

    print(true_triples[idx])
    print(true_traces[idx])
    print(pred_traces[idx])

jaccard = []
for i in range(pred_traces.shape[0]):
    jaccard.append(utils.jaccard_score(pred_traces[i],true_traces[i]))
jaccard_idx = np.array(jaccard) < 1

if METHOD == 'explaine':
        
    d = {}

    for i in pred_traces[jaccard_idx][:,:,1]:

        tup = tuple(set(i))
    
        for predicate in tup:
            
            if predicate in d:
                d[predicate] += 1
            else:
                d[predicate] = 1

elif METHOD == 'gnn_explainer':

    d = {}
    
    for triples in pred_traces:
        
        tup = tuple(set(np.array(triples)[:,1]))
        
        for predicate in tup:

            if predicate in d:
                d[predicate] += 1
            else:
                d[predicate] = 1
        
#print(f'total count: {sum(d.values())}')

sorted_counts = sorted(d.items(),key=lambda key:key[1],reverse=True)

percentage = round(100*sorted_counts[0][1] / sum(d.values()))

print(f"of incorrect predictions {sorted_counts[0][0]} was used in {percentage}% of triples")

# output = sorted(d.items(),key=lambda x:x[1],reverse=True)

# str_out = METHOD + '_' + RULE

# np.savez(os.path.join('..','data','other',f'{str_out}.npz'),x=np.array(output,dtype=object))

#print(f"Most predicted relation pairs for {METHOD} - {RULE}: {output}")

# total_count = joblib.Parallel(n_jobs=-2, verbose=20)(
#             joblib.delayed(get_count)(i,true_triples,pred_traces,ent2idx,adj_mats,num_relations=NUM_RELATIONS)
#                 for i in range(len(true_triples))
#             )

# num_triples = true_triples.shape[0]
# print(f"number of triples: {num_triples}")
# print(f"Percentage of triples using 1st degree neighbors of head or tail: {sum(total_count) / num_triples}")

# count = 0
# for tup in output:
#     if 'brother' not in tup[0] or 'parent' not in tup[0]:
#         count += tup[1]

# print(f"Percentage of triples that did not include brother or parent {count/num_triples} ")