#!/usr/bin/env python3

import os
import utils
import random as rn
import argparse
import numpy as np

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str,
    help='royalty_30k or royalty_20k')
parser.add_argument('rule',type=str,
    help='spouse,successor,...,full_data')
parser.add_argument('embedding_dim',type=int)

args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,entities,relations = utils.get_data(data,RULE)

MAX_PADDING, LONGEST_TRACE = utils.get_longest_trace(DATASET, RULE)

_, _, X_test_triples, X_test_traces = utils.train_test_split_no_unseen(
    triples,traces,longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,test_size=.25,seed=SEED)

X_test_traces = X_test_traces[:,0:LONGEST_TRACE,:]

###################################################

gnn_data = np.load(
    os.path.join('..','data','preds',DATASET,
        'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

gnn_preds = gnn_data['preds']

num_gnn_triples = X_test_traces.shape[0]
gnn_jaccard = 0.0
for i in range(num_gnn_triples):
    gnn_jaccard += utils.jaccard_score(X_test_traces[i],gnn_preds[i])
gnn_jaccard /= num_gnn_triples

gnn_precision, gnn_recall = utils.precision_recall(X_test_traces,gnn_preds)
gnn_f1 = utils.f1(gnn_precision,gnn_recall)

print(f'{DATASET} {RULE} GnnExplainer')
print(f'precision {round(gnn_precision,3)}')
print(f'recall {round(gnn_recall,3)}')
print(f'f1 {round(gnn_f1,3)}')
print(f'jaccard score: {round(gnn_jaccard,3)}')

###################################################

explaine_data = np.load(
    os.path.join('..','data','preds',DATASET,
        'explaine_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

explaine_preds = explaine_data['preds']

num_explaine_triples = X_test_traces.shape[0]
explaine_jaccard = 0.0
for i in range(num_explaine_triples):
    explaine_jaccard += utils.jaccard_score(X_test_traces[i],explaine_preds[i])
explaine_jaccard /= num_explaine_triples

explaine_precision, explaine_recall = utils.precision_recall(X_test_traces,explaine_preds)

explaine_f1 = utils.f1(explaine_precision,explaine_recall)

print(f'{DATASET} {RULE} ExplaiNE')
print(f'precision {round(explaine_precision,3)}')
print(f'recall {round(explaine_recall,3)}')
print(f'f1 {round(explaine_f1,3)}')
print(f'jaccard score: {round(explaine_jaccard,3)}')
