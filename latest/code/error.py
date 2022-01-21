#!/usr/bin/env python3

import numpy as np
import utils
import matplotlib.pyplot as plt

def error_indices_royalty(X_test_traces,pred_traces):
    
    jaccard_scores = []

    for i in range(len(X_test_traces)):

        jaccard = utils.jaccard_score(
            X_test_traces[i],
            pred_traces[i])

        jaccard_scores.append(jaccard)
    error_idx = np.array(jaccard_scores) < 1
    
    return error_idx

def predicate_frequency(error_preds):
    
    '''
    most frequently predicted predicate amongst errors
    '''
        
    count_dict = rel_counts(error_preds)

    sorted_counts = sorted(count_dict.items(),key=lambda key:key[1],reverse=True)

    percentage = round(100*sorted_counts[0][1] / sum(count_dict.values()))
    
    return sorted_counts, percentage

def rel_counts(error_preds):

    all_rels = error_preds.reshape(-1,3)[:,1]

    rels,counts = np.unique(all_rels[all_rels!='UNK_REL'], return_counts=True)

    count_dict = dict(zip(list(rels), list(counts)))

    return count_dict

def histograms(error_preds):
    
    counts = rel_counts(error_preds.reshape(-1,3))

    sorted_counts = sorted(counts.items(), key=lambda x:x[1],reverse=True)
    keys = [tup[0] for tup in sorted_counts]
    values = [tup[1] for tup in sorted_counts]
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax.bar(keys,values)
    ax.set_xticklabels(labels=keys,rotation = (65), fontsize = 12)
    #fig.tight_layout()
    plt.savefig(os.path.join('..','data','plots',
        DATASET,f"{DATASET}_{METHOD}_{RULE}_counts.pdf"),bbox_inches='tight')

# def missing_predicates(error_preds, X_test_traces):
    
#     '''most frequently missing predicate (royalty-20k,royalty-30k)'''
    
#     percents = []
#     error_preds = error_preds.flatten()

#     for unique_rel in np.unique(error_preds[:,1]):

#         if unique_rel == 'UNK_REL':
#             continue

#         percent = 100 * ((error_preds[:,1] != unique_rel).sum() / (len(error_preds)))

#         percents.append((unique_rel, percent))
        
#     sorted_percents = sorted(percents, key=lambda x:x[1],reverse=True)

#     return sorted_percents

def missing_predicates(error_preds, X_test_traces):

    '''Returns dict of predicate along with percentage of triples
    that do not include this predicate'''

    count_dict = rel_counts(error_preds)

    missing_predicates_dict = {}

    for predicate, count in count_dict.items():

        total = sum(count_dict.values())
    
        missing_predicates_dict[predicate] = round(100 * ((total - count) / total))

    return missing_predicates_dict

if __name__ == "__main__":

    import os
    import random as rn
    import argparse
    import json
    import collections

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
        help='royalty_30k, royalty_20k')
    parser.add_argument('rule',type=str,
        help='spouse,successor,...,full_data')
    parser.add_argument('method',type=str)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = PRINT_RULE = args.rule
    METHOD = args.method

    print(f"using weights from {RULE}")

    pred_data = np.load(
        os.path.join('..','data','preds', DATASET,METHOD+'_'+DATASET+'_'+RULE+'_preds.npz'),
        allow_pickle=True
    )
    pred_traces = pred_data['preds']


    DATA = np.load(os.path.join('..','data',DATASET+'.npz'))
    triples,traces,entities,relations = utils.get_data(DATA,PRINT_RULE)

    MAX_PADDING, LONGEST_TRACE = utils.get_longest_trace(DATASET,PRINT_RULE)

    _, _, X_test_triples, X_test_traces = utils.train_test_split_no_unseen(
        triples, 
        traces,
        longest_trace=LONGEST_TRACE,
        max_padding=MAX_PADDING,
        test_size=.25,
        seed=SEED)

    pred_traces = pred_traces[:,0:LONGEST_TRACE,:]
    error_idx = error_indices_royalty(X_test_traces,pred_traces)

    error_preds = pred_traces[error_idx].reshape(-1,3)

    print(DATASET)
    print(METHOD)
    print(PRINT_RULE)

    if DATASET == 'royalty_20k' or DATASET == 'royalty_30k':

        sorted_counts, percentage = predicate_frequency(error_preds)

        print(f"of incorrect predictions {sorted_counts[0][0]} was used in {percentage}% of triples")

        missing_predicates_dict = missing_predicates(error_preds, X_test_traces)

        print(f"Missing predicates {missing_predicates_dict}")

    print('Done.')