#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import random as rn
import argparse

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str,
    help='royalty_15k or royalty_20k')
parser.add_argument('rule',type=str,
    help='spouse,successor,...,full_data')
parser.add_argument('trace_length',type=int)
parser.add_argument('method',type=str, help='gnn_explainer or explaine') 
args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
TRACE_LENGTH = args.trace_length
METHOD = args.method

if DATASET == 'royalty_15k':
    d = {'spouse':1,'grandparent':2}
elif DATASET == 'royalty_20k':
    d = {'spouse':1,'successor':1,
        'predecessor':1}

for rule,trace_length in d.items():
    
    pred_data = np.load(
        os.path.join('..','data','preds',DATASET,METHOD+'_'+DATASET+'_'+rule+'_preds.npz'),
        allow_pickle=True
    )
    
    triples,traces,entities,relations = utils.get_data(data,rule)
    
    pred_traces = pred_data['preds']
    
    true_triples = triples[pred_data['test_idx']]
    true_traces = traces[pred_data['test_idx']][:,0:TRACE_LENGTH,:]
    
    jaccard = []
    for i in range(pred_traces.shape[0]):
        jaccard.append(utils.jaccard_score(pred_traces[i],true_traces[i]))
    error_idx = np.array(jaccard) < 1
    
    counts = {}

    for pred in pred_traces[error_idx]:
        for triple in pred:
            if triple[1] in counts:
                counts[triple[1]] += 1
            else:
                counts[triple[1]] = 1
                
    sorted_counts = sorted(counts.items(), key=lambda x:x[1],reverse=True)
    keys = [tup[0] for tup in sorted_counts]
    values = [tup[1] for tup in sorted_counts]
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax.bar(keys,values)
    ax.set_xticklabels(labels=keys,rotation = (45), fontsize = 10)
    
    plt.savefig(f"../data/plots/{DATASET}_{METHOD}_{rule}_counts.pdf",bbox_inches='tight')