#!/usr/bin/env python3

import numpy as np
import utils
import os
import tensorflow as tf
import joblib
import argparse
import random as rn

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str,
    help='royalty_30k or royalty_20k')
parser.add_argument('rule',type=str,
    help='spouse,successor,...,full_data')
parser.add_argument('method',type=str, help='gnn_explainer or explaine') 
args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
METHOD = args.method

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,entities,relations = utils.get_data(data,RULE)

MAX_PADDING, LONGEST_TRACE = utils.get_longest_trace(DATASET, RULE)

_, _, X_test_triples, X_test_traces = utils.train_test_split_no_unseen(
    triples,traces,longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,test_size=.25,seed=SEED)

pred_data = np.load(
    os.path.join('..','data','preds', DATASET,METHOD+'_'+DATASET+'_'+RULE+'_preds.npz'),
    allow_pickle=True
)

pred_traces = pred_data['preds']

pred_traces = pred_traces[:,0:LONGEST_TRACE,:]

true_traces = X_test_traces[:,0:LONGEST_TRACE,:]

jaccard = []
for i in range(pred_traces.shape[0]):
    jaccard.append(utils.jaccard_score(true_traces[i],pred_traces[i]))
error_idx = np.array(jaccard) < 1

d = {}

for triples in pred_traces[error_idx]:

    tup = tuple(set(np.array(triples)[:,1]))

    for predicate in tup:

        if predicate in d:
            d[predicate] += 1
        else:
            d[predicate] = 1

sorted_counts = sorted(d.items(),key=lambda key:key[1],reverse=True)

percentage = round(100*sorted_counts[0][1] / sum(d.values()))
print(DATASET)
print(RULE)
print(METHOD)
print('###########################################')

print(f"of incorrect predictions {sorted_counts[0][0]} was used in {percentage}% of triples")

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