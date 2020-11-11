#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN

def get_k_hop_adj(head,tail,adj_mats,num_relations,num_entities):

    k_hop_adj_mats = []

    for r in range(num_relations):

        k_hop_adj_mat = np.zeros((NUM_ENTITIES,NUM_ENTITIES),dtype='float32')

        head_neighbors = tf.where(adj_mats[0,r,:][head]==1.)[:,-1]
        tail_neighbors = tf.where(adj_mats[0,r,:][tail]==1.)[:,-1]

        for h_i in head_neighbors:
            k_hop_adj_mat[head,h_i] = 1.
            k_hop_adj_mat[h_i,head] = 1.
            
        for t_i in tail_neighbors:
            k_hop_adj_mat[tail,t_i] = 1.
            k_hop_adj_mat[t_i,tail] = 1.

        k_hop_adj_mats.append(k_hop_adj_mat)

    return tf.expand_dims(k_hop_adj_mats,axis=0)

def get_grads(head,rel,tail,masks,model,y_true,all_indices,k_hop_adj_mats,loss_object):

    with tf.GradientTape() as tape:

        tape.watch(masks)

        masked_adj = k_hop_adj_mats*tf.nn.sigmoid(masks)

        y_pred = model(
            [
            all_indices,
            head,
            rel,
            tail,
            masked_adj
            ]
        )
        print(y_pred)

        loss = loss_object(y_true,y_pred)

    return loss,tape.gradient(loss,masks)

def get_true_subgraph(exp2idx_i,num_relations,num_entities):

    true_subgraph = []

    for i in range(num_relations):

        mat = np.zeros((num_entities,num_entities))

        exp_triples = exp2idx_i[exp2idx_i[:,1] == i]

        for h,_,t in exp_triples:

            mat[h,t] = 1
            mat[t,h] = 1

        true_subgraph.append(mat)

    return tf.expand_dims(true_subgraph,axis=0)

def binary_jaccard(truth,pred):
    
    m11 = np.logical_and(truth==1,pred==1).sum()
    m01 = np.logical_and(truth==0,pred==1).sum()
    m10 = np.logical_and(truth==1,pred==0).sum()
    
    return (m11 / (m01+m10+m11))

if __name__ == '__main__':

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    data = np.load(os.path.join('..','data','royalty.npz'))

    entities = data['entities'].tolist()
    relations = data['relations'].tolist()

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    EMBEDDING_DIM = 30
    OUTPUT_DIM = 50
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 2

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    triples, traces = data['spouse_triples'], data['spouse_traces']

    train2idx = utils.array2idx(triples, ent2idx,rel2idx)
    trainexp2idx = utils.array2idx(traces, ent2idx,rel2idx)

    NUM_TRIPLES = train2idx.shape[0]

    adj_mats = utils.get_adjacency_matrix_list(
        num_relations=NUM_RELATIONS,
        num_entities=NUM_ENTITIES,
        data=train2idx
    )

    train2idx = np.expand_dims(train2idx,axis=0)

    all_indices = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))#np.arange(NUM_ENTITIES).reshape(1,-1)

    model = RGCN.get_RGCN_Model(
        num_triples=NUM_TRIPLES,
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )
    model.load_weights(os.path.join('..','data','weights','rgcn.h5'))

    bce = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    jaccard_scores = []

    for i in range(train2idx.shape[1]):

        head = train2idx[:,i:i+1,0]
        rel = train2idx[:,i:i+1,1]
        tail = train2idx[:,i:i+1,2]

        k_hop_adj_mats = get_k_hop_adj(head,tail,adj_mats,NUM_RELATIONS,NUM_ENTITIES)

        y_true = model([
            all_indices,
            head,
            rel,
            tail,
            k_hop_adj_mats])

        masks = tf.Variable(
            initial_value=tf.random.normal(
                (adj_mats.shape), 
                mean=0, 
                stddev=1, 
                dtype=tf.dtypes.float32, 
                seed=SEED),
            name='mask',
            trainable=True
        )

        for epoch in range(NUM_EPOCHS):

            loss_val,grads = get_grads(
                head=head,
                rel=rel,
                tail=tail,
                masks=masks,
                model=model,
                y_true=y_true,
                all_indices=all_indic   es,
                k_hop_adj_mats=k_hop_adj_mats,
                loss_object=bce
                )

            print("loss val: ",loss_val)
            optimizer.apply_gradients(zip([grads],[masks]))

        pred_subgraph = tf.cast((k_hop_adj_mats*tf.nn.sigmoid(masks)) > .5,dtype=tf.int32)
        true_subgraph = get_true_subgraph(trainexp2idx[i],NUM_RELATIONS,NUM_ENTITIES)

        print(binary_jaccard(true_subgraph,pred_subgraph)) 
        break

