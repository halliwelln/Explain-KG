#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from collections import defaultdict
from scipy import sparse

def get_negative_triples(head, rel, tail, num_entities, seed):
    
    cond = tf.random.uniform(head.shape, 0, 2, dtype=tf.int64, seed=seed) #1 means keep entity
    rnd = tf.random.uniform(head.shape, 0, num_entities-1, dtype=tf.int64, seed=seed)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)   

    return neg_head, neg_tail

def train2idx(dataset, ent2idx, rel2idx):

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

def jaccard_score(ground_truth,generated_explanation):

    scores = []

    for i in range(len(ground_truth)):

        true_set = set(ground_truth[i])
        pred_set = set(generated_explanation[i])

        intersect = len(true_set.intersection(pred_set))
        union = len(true_set.union(pred_set))

        scores.append(intersect/union)

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
