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

def array2idx(dataset, ent2idx, rel2idx):

    '''
    Convert numpy array of strings to indices
    '''

    data = []

    for head, rel, tail in dataset:

        head_idx = ent2idx[head]
        tail_idx = ent2idx[tail]
        rel_idx = rel2idx[rel]

        data.append((head_idx, rel_idx, tail_idx))

    data = np.array(data)

    return data

def idx2train(dataset, idx2ent, idx2rel):

    data = []

    for h,r,t in dataset:

        head = idx2ent[h]
        rel = idx2rel[r]
        tail = idx2ent[t]

        data.append((head,rel,tail))

    data = np.array(data)

    return data

def jaccard_score(true_exp,pred_exp):

    assert len(true_exp) == len(pred_exp)

    scores = []

    for i in range(len(true_exp)):

        pred_i = pred_exp[i]
        true_i = true_exp[i]

        num_true_traces = min(true_i.ndim,true_i.shape[0])

        if isinstance(pred_i,np.ndarray):
            num_pred_traces = pred_i.ndim
        
        elif isinstance(pred_i,list):
            num_pred_traces = len(pred_i)
    
        bool_array = (pred_i == true_i)

        count = 0

        for row in bool_array:
            if row.all():
                count +=1

        score = count / (num_true_traces+num_pred_traces-count)

        scores.append(score)

    return np.mean(scores)

    # scores = []

    # for i in range(len(ground_truth)):

    #     true_set = set(ground_truth[i])
    #     pred_set = set(generated_explanation[i])

    #     intersect = len(true_set.intersection(pred_set))
    #     union = len(true_set.union(pred_set))

    #     scores.append(intersect/union)

    # return np.mean(scores)

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

def get_neighbor_idx(A):

    A = sparse.coo_matrix(A)

    indices = {}

    for i,j in zip(A.row,A.col):

        if i in indices:
            indices[i].append(j)

        else:
            indices[i] = [j]

    return indices

def get_tup(line_str):
    
    line_str = line_str.split()[:-1]
    
    source_tup = []
    for i in line_str:

        if 'dbpedia.org/resource' in i:
            source_tup.append(i.split('/')[-1])
        else:
            source_tup.append(i.split(':')[-1])
        
    return tuple(source_tup)

def parse_traces(file_name):
    
    lines = []

    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line)

    traces = defaultdict(list)

    for idx, line in enumerate(lines):

        if "graph us:construct" in line:

            source_tup = get_tup(lines[idx+1])        

            assert len(source_tup) == 3

            traces[source_tup] = []

        if 'graph us:where' in line:

            for sub_line in lines[idx+1:]:

                if sub_line.strip() == '}':      
                    break

                exp_tup = get_tup(sub_line)
                traces[source_tup].append(exp_tup)
                assert len(exp_tup) == 3  
                
    return traces
