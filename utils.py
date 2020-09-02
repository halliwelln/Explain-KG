#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy import sparse

def get_negative_triples(head, rel, tail, num_entities, random_state):
    
    cond = tf.random.uniform(head.shape, 0, 2, dtype=tf.int64, seed=random_state) #1 means keep entity
    rnd = tf.random.uniform(head.shape, 0, num_entities-1, dtype=tf.int64, seed=random_state)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)   

    return neg_head, neg_tail

def get_entity_embeddings(model):
    '''Embedding matrix for entities'''
    return model.get_layer('entity_embeddings').get_weights()[0]

def get_relation_embeddings(model):
    '''Embedding matrix for relations'''
    return model.get_layer('relation_embeddings').get_weights()[0]

def array2idx(dataset, ent2idx,rel2idx):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head, rel, tail in dataset:
            
            head_idx = ent2idx[head]
            tail_idx = ent2idx[tail]
            rel_idx = rel2idx[rel]
            
            data.append((head_idx, rel_idx, tail_idx))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head,rel,tail in dataset[i,:,:]:

                if (head == '0.0') or (tail == '0.0') or (rel == '0.0'):
                    temp_array.append((-1,-1,-1))
                    continue

                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx,rel_idx,tail_idx))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    return data

# def idx2train(dataset, idx2ent, idx2rel):

#     data = []

#     for h,r,t in dataset:

#         head = idx2ent[h]
#         rel = idx2rel[r]
#         tail = idx2ent[t]

#         data.append((head,rel,tail))

#     data = np.array(data)

#     return data

def jaccard_score(true_exp,pred_exp):

    assert len(true_exp) == len(pred_exp)

    scores = []

    for i in range(len(true_exp)):

        pred_i = pred_exp[i]
        true_i = true_exp[i]

        if isinstance(true_i,np.ndarray):
            num_true_traces = min(true_i.ndim,true_i.shape[0])

        elif isinstance(true_i,list):
            num_true_traces = len(true_i)

        if isinstance(pred_i,np.ndarray):
            num_pred_traces = min(pred_i.ndim,pred_i.shape[0])
        
        elif isinstance(pred_i,list):
            num_pred_traces = len(pred_i)
    
        bool_array = (pred_i == true_i).reshape(num_true_traces,3)

        count = 0

        for row in bool_array:
            if row.all():
                count +=1

        score = count / (num_true_traces+num_pred_traces-count)

        scores.append(score)

    return np.mean(scores)

def get_adjacency_matrix(data,entities,num_entities):

    row = []
    col = []

    for h,r,t in data:

        h_idx = entities.index(h)
        t_idx = entities.index(t)

        row.append(h_idx)
        col.append(t_idx)

    adj = np.ones(len(row))

    return sparse.csr_matrix((adj,(row,col)),shape=(num_entities,num_entities))

def get_tup(line_str):
    
    line_str = line_str.split()[:-1]
    
    source_tup = []
    for i in line_str:

        if 'dbpedia.org/resource' in i:
            source_tup.append(i.split('resource/')[-1])
        else:
            source_tup.append(i.split(':')[-1])
        
    return list(source_tup)

def parse_ttl(file_name,max_traces):
    
    lines = []

    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line)

    ground_truth = []
    traces = []

    for idx in range(len(lines)):

        if "graph us:construct" in lines[idx]:

            source_tup = get_tup(lines[idx+1])

        exp_triples = []

        if 'graph us:where' in lines[idx]:

            while lines[idx+1] != '} \n':

                exp_tup = get_tup(lines[idx+1])
                exp_triples.append(exp_tup)

                idx+=1

        if len(source_tup) != 0 and len(exp_triples) != 0:
            
            no_name_entity = False
            
            if ("no_name_entry" in source_tup[0]) or ("no_name_entry" in source_tup[2]):
                no_name_entity = True
            
            for h,r,t in exp_triples:
                if ("no_name_entry" in h) or ("no_name_entry" in t):
                    no_name_entity = True
            
            if not no_name_entity:

                if len(exp_triples) < max_traces:
                    
                    while len(exp_triples) != max_traces:
                        
                        pad = np.zeros((3))
                        exp_triples.append(pad)

                ground_truth.append(np.array(source_tup))
                traces.append(np.array(exp_triples))

    return np.array(ground_truth), np.array(traces)
