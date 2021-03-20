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

    all_preds = pred_data['preds']

    for i in range(len(all_preds)):

        preds_i = []

        for rel_idx in range(NUM_RELATIONS):

            triples_i = all_preds[i][rel_idx]

            if triples_i.shape[0]:
                rel_indices = (np.ones((triples_i.shape[0],1)) * rel_idx).astype(np.int64)
                concat = np.concatenate([triples_i,rel_indices],axis=1)
                preds_i.append(concat[:,[0,2,1]])
        preds_i = np.concatenate(preds_i,axis=0)
        pred_traces.append(utils.idx2array(preds_i,idx2ent,idx2rel))

    pred_traces = np.array(pred_traces,dtype=object)

true_triples = triples[pred_data['test_idx']]
true_traces = traces[pred_data['test_idx']][:,0:TRACE_LENGTH,:]

jaccard = []
for i in range(pred_traces.shape[0]):
    jaccard.append(utils.jaccard_score(pred_traces[i],true_traces[i]))
error_idx = np.array(jaccard) < 1

if RULE == 'spouse' and METHOD == 'explaine':
    
    errors = pred_traces[error_idx]
        
    idx = np.argwhere((np.array(jaccard) < 1) & (pred_traces[:,0,:][:,1] == 'spouse'))[-25]

    print(true_triples[idx])
    print(true_traces[idx])
    print(pred_traces[idx])

if METHOD == 'explaine':
        
    d = {}

    for i in pred_traces[error_idx][:,:,1]:

        tup = tuple(set(i))
    
        for predicate in tup:
            
            if predicate in d:
                d[predicate] += 1
            else:
                d[predicate] = 1

elif METHOD == 'gnn_explainer':

    d = {}
    
    for triples in pred_traces[error_idx]:
        
        tup = tuple(set(np.array(triples)[:,1]))
        
        for predicate in tup:

            if predicate in d:
                d[predicate] += 1
            else:
                d[predicate] = 1
        
#print(f'total count: {sum(d.values())}')

sorted_counts = sorted(d.items(),key=lambda key:key[1],reverse=True)

percentage = round(100*sorted_counts[0][1] / sum(d.values()))

print(RULE)
print(METHOD)
print('###########################################')

print(f"of incorrect predictions {sorted_counts[0][0]} was used in {percentage}% of triples")

print('###########################################')
if METHOD == 'explaine':
    num_errors = len(pred_traces[error_idx])

    for predicate in list(true_traces[:,:,1][0]):
        
        percent = (np.sum(
            pred_traces[error_idx][:,:,1]!=predicate,
            axis=1) >= 1).sum() / num_errors
        
        print(f"{predicate} missing from {round(percent*100)}% of errors ")
else:
    indicator = []

    for pred in pred_traces[error_idx]:
        
        counts = {}
        
        for triple in pred:
            if triple[1] in counts:
                counts[triple[1]] += 1
            else:
                counts[triple[1]] = 1
        
        indicator_i = 0.0
        for predicate in list(true_traces[:,:,1][0]):
            
            for p,v in counts.items():
                
                if predicate == p and v >= 1:
                    
                    indicator_i += 1
                    
        if indicator_i >= len(list(true_traces[:,:,1][0])):
            indicator.append(1)
                    
    percent = 1-sum(indicator) / len(pred_traces[error_idx])

    print(f"{predicate} missing from {round(percent*100)}% of errors ")
