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

def train_test_split_no_unseen(
    X,
    E,
    unk_ent_id='UNK_ENT',
    unk_rel_id='UNK_REL',
    test_size=.3,
    seed=123,
    allow_duplication=False):

    test_size = int(len(X) * test_size)

    np.random.seed(seed)

    X_train = None
    X_train_exp = None
    X_test_candidates = X
    X_test_exp_candidates = E

    exp_entities = np.array([
        [E[:,i,j,0],E[:,i,j,2]] for i in range(LONGEST_TRACE) for j in range(MAX_PADDING)]).flatten()

    exp_relations = np.array([
        [E[:,i,j,1]] for i in range(LONGEST_TRACE) for j in range(MAX_PADDING)]).flatten()

    entities, entity_cnt = np.unique(np.concatenate([
                                triples[:,0], triples[:,2], exp_entities],axis=0),return_counts=True)
    rels, rels_cnt = np.unique(np.concatenate([
                                triples[:,1], exp_relations],axis=0),return_counts=True)
    
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []
    
    all_indices_shuffled = np.random.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        test_exp = utils.remove_padding_np(X_test_exp_candidates[idx],unk_ent_id, unk_rel_id,axis=-1)
                
        # reduce the entity and rel count of triple
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1
        
        exp_entities = np.concatenate([test_exp[:,0].flatten(),
                                       test_exp[:,2].flatten()])
        
        exp_rels = test_exp[:,1]
        
        # reduce the entity and rel count of explanation
        for exp_ent in exp_entities:
            dict_entities[exp_ent] -= 1
            
        for exp_rel in exp_rels:
            dict_rels[exp_rel] -= 1
            
        ent_counts = []
        for exp_ent in exp_entities:
            count_i = dict_entities[exp_ent]
            
            if count_i > 0:
                ent_counts.append(1)
            else:
                ent_counts.append(0)
                
        rel_counts = []
        for exp_rel in exp_rels:
            count_i = dict_rels[exp_rel]
            
            if count_i > 0:
                rel_counts.append(1)
            else:
                rel_counts.append(0)
        
        #compute sums and determine if counts > 0

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0 and \
                sum(ent_counts) == len(ent_counts) and \
                sum(rel_counts) == len(rel_counts):
            
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))
                
                # break out of the loop
                break
            
        else:
            # since removing this triple results in unseen entities, add it to training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            
            for exp_ent in exp_entities:
                dict_entities[exp_ent] += 1
            
            for exp_rel in exp_rels:
                dict_rels[exp_rel] += 1
            
            idx_train.append(idx)
            
    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = np.random.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating 
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True," 
                            "or set test_size to a smaller value.")

    X_train = X_test_candidates[idx_train]
    X_train_exp = X_test_exp_candidates[idx_train]
    
    X_test = X_test_candidates[idx_test]
    X_test_exp = X_test_exp_candidates[idx_test]
    
    #shuffle data
    
    idx_train_shuffle = np.random.permutation(np.arange(len(idx_train)))
    idx_test_shuffle = np.random.permutation(np.arange(len(idx_test)))
    
    X_train = X_train[idx_train_shuffle]
    X_train_exp = X_train_exp[idx_train_shuffle]
    
    X_test = X_test[idx_test_shuffle]
    X_test_exp = X_test_exp[idx_test_shuffle]
    
    return X_train, X_train_exp, X_test, X_test_exp

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
