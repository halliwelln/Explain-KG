#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from scipy import sparse

def jaccard_score(true_exp,pred_exp):

    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count +=1

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

# def get_data(data,rule):

#     if rule == 'full_data':
#         triples,traces,nopred = concat_triples(data, data['rules'])
#         entities = data['all_entities'].tolist()
#         relations = data['all_relations'].tolist()
#     elif rule != 'grandparent':
#         triples,traces,nopred = concat_triples(data, [rule,'brother','sister'])
#         sister_relations = data['sister_relations'].tolist()
#         sister_entities = data['sister_entities'].tolist()

#         brother_relations = data['brother_relations'].tolist()
#         brother_entities = data['brother_entities'].tolist()

#         entities = np.unique(data[rule + '_entities'].tolist()+brother_entities+sister_entities).tolist()
#         relations = np.unique(data[rule + '_relations'].tolist()+brother_relations+sister_relations).tolist()

#     elif rule == 'grandparent':
#         triples,traces = data[rule + '_triples'],data[rule + '_traces']
#         entities = data[rule+'_entities'].tolist()
#         relations = data[rule+'_relations'].tolist()
#         nopred = []

#     return triples,traces,nopred,entities,relations

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

def train_test_split_no_unseen(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):

    '''Taken from https://github.com/Accenture/AmpliGraph/blob/master/ampligraph/evaluation/protocol.py'''

    if type(test_size) is float:
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    subs, subs_cnt = np.unique(X[:, 0], return_counts=True)
    objs, objs_cnt = np.unique(X[:, 2], return_counts=True)
    rels, rels_cnt = np.unique(X[:, 1], return_counts=True)
    dict_subs = dict(zip(subs, subs_cnt))
    dict_objs = dict(zip(objs, objs_cnt))
    dict_rels = dict(zip(rels, rels_cnt))

    idx_test = np.array([], dtype=int)

    loop_count = 0
    tolerance = len(X) * 10

    if filtered_test_predicates:
        test_triples_idx = np.where(np.isin(X[:, 1], filtered_test_predicates))[0]
    else:
        test_triples_idx = np.arange(len(X))

    while idx_test.shape[0] < test_size:
        i = rnd.choice(test_triples_idx)
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i, 1]] > 1:
            dict_subs[X[i, 0]] -= 1
            dict_objs[X[i, 2]] -= 1
            dict_rels[X[i, 1]] -= 1
            if allow_duplication:
                idx_test = np.append(idx_test, i)
            else:
                idx_test = np.unique(np.append(idx_test, i))

        loop_count += 1

        if loop_count == tolerance:
            if allow_duplication:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Change seed values, remove filter on test predicates or set "
                                "test_size to a smaller value.")
            else:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Set allow_duplication=True,"
                                "change seed values, remove filter on test predicates or "
                                "set test_size to a smaller value.")
    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)

    return idx_train,idx_test

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

# def get_adj_mats_list(num_relations,num_entities,data):

#     adj_mats = []

#     for i in range(num_relations):

#         adj = np.zeros((num_entities,num_entities))

#         for h,_,t in (data[data[:,1] == i]):

#             adj[h,t] = 1
#             adj[t,h] = 1

#         adj_mats.append(adj)

#     return np.expand_dims(adj_mats,axis=0)

# def get_adjacency_matrix(data,num_entities):

#     row = []
#     col = []

#     for h,_,t in data:

#         row.append(h)
#         col.append(t)

#     adj = np.ones(len(row))

#     return sparse.csr_matrix((adj,(row,col)),shape=(num_entities,num_entities))

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

# def concat_triples(data, rules):

#     triples = []
#     traces = []
#     no_pred_triples = []
#     no_pred_traces = []

#     for rule in rules:

#         triple_name = rule + '_triples'
#         traces_name = rule + '_traces'

#         if ('brother' in rule) or ('sister' in rule):
#             no_pred_triples.append(data[triple_name])
#             no_pred_traces.append(data[traces_name])
#         else:
#             triples.append(data[triple_name])
#             traces.append(data[traces_name])

#     triples = np.concatenate(triples, axis=0)
#     traces = np.concatenate(traces, axis=0)
#     no_pred_triples = np.concatenate(no_pred_triples, axis=0)
#     no_pred_traces = np.concatenate(no_pred_traces, axis=0)

#     no_pred = np.concatenate([no_pred_triples,no_pred_traces.reshape(-1,3)],axis=0)
    
#     return triples, traces, no_pred

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

# def idx2train(dataset, idx2ent, idx2rel):

#     data = []

#     for h,r,t in dataset:

#         head = idx2ent[h]
#         rel = idx2rel[r]
#         tail = idx2ent[t]

#         data.append((head,rel,tail))

#     data = np.array(data)

#     return data

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
