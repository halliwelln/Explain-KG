#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def f1(precision,recall):
    return 2 * (precision*recall) / (precision + recall)

def jaccard_score(true_exp,pred_exp):

    rels = set(true_exp[:,1])
    if ('spouse' in rels) or ('successor' in rels) or ('predecessor' in rels):
        true_exp = true_exp[0:1,:]
        
    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count +=1

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def precision_recall(true_exps,preds):

    num_triples = true_exps.shape[0]

    precision = 0.0
    recall = 0.0

    for i in range(num_triples):
        
        current_tp = 0.0
        current_fp = 0.0
        current_fn = 0.0
        
        true_exp = true_exps[i]
        current_preds = preds[i]

        rels = set(true_exp[:,1])
        #remove padding triple
        if ('spouse' in rels) or ('successor' in rels) or ('predecessor' in rels):
            true_exp = true_exp[0:1,:]

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

def get_data(data,rule):

    if rule == 'full_data':

        triples,traces = concat_triples(data, data['rules'])
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples,traces = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,entities,relations

def train_test_split_no_unseen(triples,traces,test_size=.3,seed=123):

    train_triples = []
    train_traces = []
    test_triples = []
    test_traces = []

    seen_ents = {'UNK_ENT'}
    seen_rels = {'UNK_REL'}

    for i in range(len(triples)):
        
        triple = triples[i]
        trace = traces[i]
        
        if trace.ndim == 2:
        
            ents = np.unique(np.concatenate([[triple[0], triple[2]],trace[:,0],trace[:,2]]))
            rels = np.unique(np.concatenate([[triple[1]],trace[:,1]]))
            
        elif trace.ndim == 3:
            ents = np.unique(np.concatenate([
                [triple[0], triple[2]],
                trace[:,:,0].flatten(),
                trace[:,:,2].flatten()
            ]))
            
            rels = np.unique(np.concatenate([[triple[1]],trace[:,:,1].flatten()]))
        
        num_ents = len(ents)
        num_rels = len(rels)
        
        ent_count = 0
        for ent in ents:
            if ent in seen_ents:
                ent_count += 1

        rel_count = 0        
        for rel in rels:
            if rel in seen_rels:
                rel_count += 1

        if (num_ents == ent_count) and (num_rels == rel_count):
            test_triples.append(triple)
            test_traces.append(trace)
        else:
            
            train_triples.append(triple)
            train_traces.append(trace)
            
            seen_ents.update(ents)
            seen_rels.update(rels)

    rnd = np.random.RandomState(seed)

    train_triples = np.array(train_triples)
    train_traces = np.array(train_traces)

    test_triples = np.array(test_triples)
    test_traces = np.array(test_traces)

    idx = int(len(triples) * test_size)

    out_test_triples = test_triples[0:idx]
    out_test_traces = test_traces[0:idx]

    to_train_triples = test_triples[idx:]
    to_train_traces = test_traces[idx:]

    out_train_triples = np.concatenate([train_triples,to_train_triples],axis=0)
    out_train_traces = np.concatenate([train_traces,to_train_traces],axis=0)

    train_indices = rnd.permutation(np.arange(len(out_train_triples)))
    test_indices = rnd.permutation(np.arange(len(out_test_triples)))

    X_train_triples = out_train_triples[train_indices]
    X_train_traces = out_train_traces[train_indices]

    X_test_triples = out_test_triples[test_indices]
    X_test_traces = out_test_traces[test_indices]

    return X_train_triples, X_train_traces, X_test_triples, X_test_traces

def distinct(a):
    _a = np.unique(a,axis=0)
    return _a

def get_adj_mats(data,num_entities,num_relations):

    adj_mats = []

    for i in range(num_relations):

        data_i = data[data[:,1] == i]

        if not data_i.shape[0]:
            indices = tf.zeros((1,2),dtype=tf.int64)
            values = tf.zeros((indices.shape[0]))

        else:

            # indices = tf.concat([
            #         tf.gather(data_i,[0,2],axis=1),
            #         tf.gather(data_i,[2,0],axis=1)],axis=0)
            indices = tf.gather(data_i,[0,2],axis=1)

            indices = tf.py_function(distinct,[indices],indices.dtype)
            values = tf.ones((indices.shape[0]))

        sparse_mat = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(num_entities,num_entities)
            )

        sparse_mat = tf.sparse.reorder(sparse_mat)

        sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1,num_entities,num_entities))

        adj_mats.append(sparse_mat)

    return adj_mats

def get_negative_triples(head, rel, tail, num_entities, random_state=123):

    #cond = tf.random.uniform(head.shape, 0, 2, dtype=tf.int64, seed=random_state) #1 means keep entity
    #rnd = tf.random.uniform(head.shape, 0, num_entities-1, dtype=tf.int64, seed=random_state)

    cond = tf.random.uniform(tf.shape(head), 0, 2, dtype=tf.int64, seed=random_state)
    rnd = tf.random.uniform(tf.shape(head), 0, num_entities-1, dtype=tf.int64, seed=random_state)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)

    return neg_head, neg_tail

def concat_triples(data, rules):

    triples = []
    traces = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'

        triples.append(data[triple_name])
        traces.append(data[traces_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    
    return triples, traces

def array2idx(dataset,ent2idx,rel2idx):
    
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

                # if (head == '0.0') or (tail == '0.0') or (rel == '0.0'):
                #     temp_array.append((-1,-1,-1))
                #     continue

                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx,rel_idx,tail_idx))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    return data

def idx2array(dataset,idx2ent,idx2rel):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head_idx, rel_idx, tail_idx in dataset:
            
            head = idx2ent[head_idx]
            tail = idx2ent[tail_idx]
            rel = idx2rel[rel_idx]
            
            data.append((head, rel, tail))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head_idx, rel_idx, tail_idx in dataset[i,:,:]:

                # if (head == '0.0') or (tail == '0.0') or (rel == '0.0'):
                #     temp_array.append((-1,-1,-1))
                #     continue

                head = idx2ent[head_idx]
                tail = idx2ent[tail_idx]
                rel = idx2rel[rel_idx]

                temp_array.append((head,rel,tail))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    return data

def get_tup(line_str):
    
    line_str = line_str.split()[:-1]
    
    source_tup = []
    for i in line_str:

        if 'dbpedia.org/resource' in i:
            source_tup.append(i.split('resource/')[-1])
        else:
            source_tup.append(i.split(':')[-1])
        
    return list(source_tup)

def parse_ttl(file_name,max_padding):
    
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

                if len(exp_triples) < max_padding:
                    
                    while len(exp_triples) != max_padding:
                        
                        pad = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
                        exp_triples.append(pad)

                ground_truth.append(np.array(source_tup))
                traces.append(np.array(exp_triples))

    return np.array(ground_truth), np.array(traces)
