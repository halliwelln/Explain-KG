#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN

def get_computation_graph(head,rel,tail,data,num_relations):
    
    '''Get 1st degree neighbors of head and tail'''
     
    subset = data[data[:,1] == rel]

    neighbors_head = tf.concat([data[data[:,0] == head],
                                data[data[:,2] == head]],axis=0)
    neighbors_tail = tf.concat([data[(data[:,0] == tail) & (data[:,0] != head)],
                                data[(data[:,2] == tail) & (data[:,0] != head)]],axis=0)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    return all_neighbors

def gnn_explainer_grads(
    head,
    rel,
    tail,
    masks,
    adj_mats,
    model,
    all_indices,
    num_relations
    ):
    
    with tf.GradientTape() as tape:

        tape.watch(masks)

        masked_adj = []

        for i in range(num_relations):
            masked_adj.append(adj_mats[i] * tf.nn.sigmoid(masks[i]))

        y_pred = model(
            [
            all_indices,
            head,
            rel,
            tail,
            masked_adj
            ]
        )

        loss = -1 * tf.math.log(y_pred + .00001) + tf.reduce_mean(tf.nn.sigmoid(masks))

    return loss, tape.gradient(loss,masks)

def score_subgraphs(
    true_subgraphs,
    adj_mats,
    masks,
    num_relations,
    num_entities
    ):
    
    '''Compute jaccard score across all relations for one triple'''

    scores = []

    for i in range(num_relations):

        mask_i = adj_mats[i] * masks[i]

        non_masked_indices = mask_i.indices[mask_i.values > .5]

        pred_graph = tf.sparse.SparseTensor(
            indices=non_masked_indices,
            values=tf.ones(non_masked_indices.shape[0]),
            dense_shape=(num_entities,num_entities)
            )

        pred_graph = tf.sparse.to_dense(pred_graph)

        true_graph = true_subgraphs[i]

        score = utils.tf_binary_jaccard(true_graph,pred_graph)

        if tf.math.is_nan(score):
            scores.append(0)
        else:
            scores.append(score)

    return tf.reduce_mean(scores)

if __name__ == '__main__':

    import argparse

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    # parser = argparse.ArgumentParser()

    # parser.add_argument('rule',type=str,help=
    #     'Enter which rule to use spouse,successor,...etc (str), -1 (str) for full dataset')
    # args = parser.parse_args()

    # RULE = args.rule

    RULE = 'spouse'

    data = np.load(os.path.join('..','data','royalty.npz'))

    if RULE == '-1':
        triples, traces,no_pred_triples,no_pred_traces = utils.concat_triples(data, data['rules'])
        RULE = 'full_data'
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    else:
        triples, traces = data[RULE + '_triples'], data[RULE + '_traces']
        entities = data[RULE + '_entities'].tolist()
        relations = data[RULE + '_relations'].tolist()  

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    EMBEDDING_DIM = 25
    OUTPUT_DIM = 50
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 2

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    train2idx = tf.convert_to_tensor(utils.array2idx(triples, ent2idx,rel2idx))
    trainexp2idx = tf.convert_to_tensor(utils.array2idx(traces, ent2idx,rel2idx))

    NUM_TRIPLES = train2idx.shape[0]

    # adj_mats = utils.get_adjacency_matrix_list(
    #     num_relations=NUM_RELATIONS,
    #     num_entities=NUM_ENTITIES,
    #     data=train2idx
    # )

    #train2idx = np.expand_dims(train2idx,axis=0)

    all_indices = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))
    #all_indices = np.arange(NUM_ENTITIES).reshape(1,-1)

    model = RGCN.get_RGCN_Model(
        num_triples=NUM_TRIPLES,
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    #model.load_weights(os.path.join('..','data','weights','rgcn.h5'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    
    #DEFINE JACCARD SCORES LIST
    #LOOP THROUGH DATA

    head = train2idx[0,0]
    rel = train2idx[0,1]
    tail = train2idx[0,2]

    true_subgraphs = utils.get_adj_mats(trainexp2idx[0],NUM_ENTITIES,NUM_RELATIONS,reshape=False)

    masks = [tf.Variable(
            initial_value=tf.random.normal(
                (NUM_ENTITIES,NUM_ENTITIES), 
                mean=0, 
                stddev=1, 
                dtype=tf.dtypes.float32, 
                seed=SEED),
            name='mask_'+str(i),
            trainable=True) for i in range(NUM_RELATIONS)
    ]

    comp_graph = get_computation_graph(head,rel,tail,train2idx,NUM_RELATIONS)
    adj_mats = utils.get_adj_mats(comp_graph, NUM_ENTITIES, NUM_RELATIONS,reshape=False)

    for epoch in range(NUM_EPOCHS):

        loss, grads = gnn_explainer_grads(
            tf.reshape(head,shape=(-1,1)),
            tf.reshape(rel,shape=(-1,1)),
            tf.reshape(tail,shape=(-1,1)),
            masks,
            adj_mats,
            model,
            all_indices,
            NUM_RELATIONS
        )

        print(f"Loss {tf.squeeze(loss).numpy()} @ epoch {epoch}")

        optimizer.apply_gradients(zip(grads,masks))


    score = score_subgraphs(
        true_subgraphs,
        adj_mats,
        masks,
        NUM_RELATIONS,
        NUM_ENTITIES
    )
    #scores.append(score)

    print(f"score {score.numpy()}")


