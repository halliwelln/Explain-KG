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
    help='royalty_20k or royalty_30k')
parser.add_argument('method',type=str, help='gnn_explainer or explaine') 
args = parser.parse_args()

DATASET = args.dataset
METHOD = args.method

if DATASET == 'royalty_30k':
    d = {'grandparent':2,'full_data':2}
elif DATASET == 'royalty_20k':
    d = {'successor':1,
        'predecessor':1,'full_data':2}

for rule,trace_length in d.items():

    data = np.load(os.path.join('..','data',DATASET+'.npz'))
    
    pred_data = np.load(
        os.path.join('..','data','preds',DATASET,METHOD+'_'+DATASET+'_'+rule+'_preds.npz'),
        allow_pickle=True
    )
    
    triples,traces,entities,relations = utils.get_data(data,rule)
    
    pred_traces = pred_data['preds']
    
    true_triples = triples[pred_data['test_idx']]
    true_traces = traces[pred_data['test_idx']][:,0:trace_length,:]
    
    jaccard = []
    for i in range(pred_traces.shape[0]):
        jaccard.append(utils.jaccard_score(true_traces[i],pred_traces[i]))
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
    ax.set_xticklabels(labels=keys,rotation = (45), fontsize = 14)
    #fig.tight_layout()
    plt.savefig(f"../data/plots/{DATASET}_{METHOD}_{rule}_counts.pdf",bbox_inches='tight')
