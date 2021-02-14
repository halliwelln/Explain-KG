#!/usr/bin/env python3

import numpy as np
import os
import utils
import random as rn
import argparse

def eval(true_exps,preds,num_triples):

    precision = 0.0
    recall = 0.0

    for i in range(num_triples):
        
        current_tp = 0.0
        current_fp = 0.0
        current_fn = 0.0
        
        true_exp = true_exps[i]
        current_preds = preds[i]

        for pred_row in current_preds:
            
            for true_row in true_exp:
                
                reversed_row = true_row[[2,1,0]]
                
                if (pred_row == true_row).all() or (pred_row == reversed_row).all():
                    current_tp += 1
                else:
                    current_fp += 1
                    
                if (current_preds == true_row).all(axis=1).sum() >= 1:
                    #if true explanation triple is in set of predicitons
                    pass
                else:
                    current_fn += 1

        if current_tp == 0 and current_fp == 0:
            current_precision = 0.0
        else:
            current_precision = current_tp / (current_tp + current_fp)

        if current_tp == 0  and current_fn == 0:
            current_recall = 0.0
        else:
            current_recall = current_tp / (current_tp + current_fn)
        
        precision += current_precision
        recall += current_recall
        
    precision /= num_triples
    recall /= num_triples

    return precision, recall

def get_true_exps(exp2idx,num_triples,trace_length):

    true_exps = []
    for i in range(num_triples):
        
        true_exps.append(exp2idx[i][0:trace_length,:])

    true_exps = np.array(true_exps)

    return true_exps

if __name__ == "__main__":

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('rule',type=str,help=
        'Enter which rule to use spouse,successor,...etc (str), full_data for full dataset')
    parser.add_argument('trace_len',type=int)
    args = parser.parse_args()

    RULE = args.rule
    TRACE_LENGTH = args.trace_len

    data = np.load(os.path.join('..','data','royalty.npz'))

    triples,traces,nopred,entities,relations = utils.get_data(data,RULE)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    EMBEDDING_DIM = 50
    OUTPUT_DIM = 50

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    if RULE != 'full_data':

        gnn_data = np.load(os.path.join('..','data','preds','gnn_explainer_'+RULE+'_preds.npz'),allow_pickle=True)

        gnn_exp2idx = utils.array2idx(traces[gnn_data['test_idx']],ent2idx,rel2idx)
        gnn_num_triples = gnn_exp2idx.shape[0]
        
        all_gnn_preds = gnn_data['preds']
        
        gnn_true_exps = get_true_exps(gnn_exp2idx,gnn_num_triples, TRACE_LENGTH)

        gnn_preds = []
        for i in range(all_gnn_preds.shape[0]):
            preds_i = []
            for idx, j in enumerate(all_gnn_preds[i]):
                if j.shape[0] > 0:
                    rel = np.ones((j.shape[0]),dtype=np.int64) * idx
                    preds_i.append(np.column_stack((j[:,0],rel,j[:,1])))            
            gnn_preds.append(np.concatenate(preds_i, axis=0))

        gnn_precision, gnn_recall = eval(gnn_true_exps,gnn_preds,gnn_num_triples)
        print(f"GnnExplainer precision {gnn_precision}, GnnExplainer recall {gnn_recall}")

    explaine_data = np.load(os.path.join('..','data','preds','explaine_'+RULE+'_preds.npz'),allow_pickle=True)

    explaine_exp2idx = utils.array2idx(traces[explaine_data['test_idx']],ent2idx,rel2idx)
    explaine_num_triples = explaine_exp2idx.shape[0]

    explaine_true_exps = get_true_exps(explaine_exp2idx,explaine_num_triples, TRACE_LENGTH)

    explaine_precision, explaine_recall = eval(explaine_true_exps,explaine_data['preds'],explaine_num_triples)
    print(f"explaiNE precision {explaine_precision}, explaiNE recall {explaine_recall}")

