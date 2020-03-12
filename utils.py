#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def get_negative_triples(head, rel, tail, num_entities):
    
    cond = tf.random.uniform(head.shape, 0, 2, dtype=tf.int64) #1 means keep entity
    rnd = tf.random.uniform(head.shape, 0, num_entities-1, dtype=tf.int64)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)   
    
    return neg_head, neg_tail

def triple2idx(dataset, ent2idx, rel2idx):

    '''
    Convert numpy array of strings to indices
    '''

    data2idx = []

    for head, rel, tail in dataset:

        head_idx = ent2idx[head]
        tail_idx = ent2idx[tail]
        rel_idx = rel2idx[rel]

        data2idx.append((head_idx, rel_idx, tail_idx))

    data2idx = np.array(data2idx)

    return data2idx