#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import random as rn

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

d = {'spouse':1,'uncle':2,
    'aunt':2,'successor':1,
    'predecessor':1,'grandparent':2}

data = np.load(os.path.join('..','data','royalty.npz'))

MODEL = 'gnn_explainer'

for rule,trace_length in d.items():

    pred_data = np.load(os.path.join('..','data','preds',MODEL+'_'+rule+'_preds.npz'),allow_pickle=True)

    triples,traces,nopred,entities,relations = utils.get_data(data,rule)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    if MODEL == 'gnn_explainer':

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

    else:
        pred_traces = utils.idx2array(pred_data['preds'],idx2ent,idx2rel)

    true_triples = triples[pred_data['test_idx']]
    true_traces = traces[pred_data['test_idx']][:,0:trace_length,:]

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
    # rects = ax.patches
    # count_labels = [str(v) for v in values]
    # for rect, label in zip(rects,count_labels):
    #     height = rect.get_height()
    #     ax.text(rect.get_x() + rect.get_width() / 2, 
    #             height,label,ha='center',va='bottom')
    plt.savefig(f"../data/plots/{MODEL}_{rule}_counts.pdf",bbox_inches='tight')

