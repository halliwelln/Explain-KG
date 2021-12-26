#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN

def get_neighbors(data_subset,node_idx):

    head_neighbors = tf.boolean_mask(data_subset,data_subset[:,0]==node_idx)
    tail_neighbors = tf.boolean_mask(data_subset,data_subset[:,2]==node_idx)

    neighbors = tf.concat([head_neighbors,tail_neighbors],axis=0)
    
    return neighbors

def get_computation_graph(head,rel,tail,data,num_relations):

    '''Get 1 hop neighbors (may include duplicates)'''

    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    return all_neighbors

def tf_jaccard(true_exp,pred_exp):

    num_true_traces = tf.shape(true_exp)[0]
    num_pred_traces = tf.shape(pred_exp)[0]

    count = 0
    for i in range(num_pred_traces):

        pred_row = pred_exp[i]

        for j in range(num_true_traces):

            true_row = true_exp[j]

            count += tf.cond(tf.reduce_all(pred_row == true_row), lambda :1, lambda:0)

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def replica_step(head,rel,tail,explanation,num_entities,num_relations):
    
    comp_graph = get_computation_graph(head,rel,tail,ADJACENCY_DATA,num_relations)

    adj_mats = utils.get_adj_mats(comp_graph, num_entities, num_relations)

    true_subgraphs = utils.get_adj_mats(tf.squeeze(explanation,axis=0),num_entities,num_relations)

    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):

        with tf.GradientTape(watch_accessed_variables=False) as tape:

            tape.watch(masks)

            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]

            before_pred = model([
                    ALL_INDICES,
                    tf.reshape(head,(1,-1)),
                    tf.reshape(rel,(1,-1)),
                    tf.reshape(tail,(1,-1)),
                    adj_mats
                    ]
                )

            pred = model([
                    ALL_INDICES,
                    tf.reshape(head,(1,-1)),
                    tf.reshape(rel,(1,-1)),
                    tf.reshape(tail,(1,-1)),
                    masked_adjs
                    ]
                )

            #penalty = [tf.reduce_sum(tf.cast(tf.sigmoid(i.values) > .5,dtype=tf.float32)) for i in masked_adjs]

            #loss = -1 * tf.math.log(pred+0.00001) + (0.0001 * tf.reduce_sum(penalty))

            loss = - before_pred * tf.math.log(pred+0.00001)

            tf.print(f"current loss {loss}")

            total_loss += loss

        grads = tape.gradient(loss,masks)
        optimizer.apply_gradients(zip(grads,masks))

    current_pred = []

    current_scores = []

    # for i in range(num_relations):

    #     mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

    #     non_masked_indices = mask_i.indices[mask_i.values > THRESHOLD]

    #     pred = non_masked_indices.numpy()

    #     current_preds.append(pred[:,1:])
    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

        mask_idx = mask_i.values > THRESHOLD

        non_masked_indices = tf.gather(mask_i.indices[mask_idx],[1,2],axis=1)

        if tf.reduce_sum(non_masked_indices) != 0:

            rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0],1)) * i,tf.int64)

            triple = tf.concat([non_masked_indices,rel_indices],axis=1)
            
            triple = tf.gather(triple,[0,2,1],axis=1)

            score_array = mask_i.values[mask_idx] 

            current_pred.append(triple)
            current_scores.append(score_array)

    current_scores = tf.concat([array for array in current_scores],axis=0)
    top_k_scores = tf.argsort(current_scores,direction='DESCENDING')[0:2]

    pred_exp = tf.reshape(tf.concat([array for array in current_pred],axis=0),(-1,3))
    pred_exp = tf.gather(pred_exp,top_k_scores,axis=0)

    for mask in masks:
        mask.assign(value=init_value)

    return total_loss, pred_exp

def distributed_replica_step(head,rel,tail,explanation,num_entities,num_relations):

    per_replica_losses, current_preds = strategy.run(replica_step,
        args=(head,rel,tail,explanation,num_entities,num_relations))

    reduce_loss = per_replica_losses / NUM_EPOCHS

    return reduce_loss, current_preds

if __name__ == '__main__':

    import argparse
    from IPython.core.debugger import set_trace
    import tensorflow as tf

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='royalty_30k or royalty_20k')
    parser.add_argument('rule',type=str,
        help='spouse,successor,...,full_data')
    parser.add_argument('num_epochs',type=int)
    parser.add_argument('embedding_dim',type=int)
    parser.add_argument('learning_rate',type=float)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = args.rule
    NUM_EPOCHS = args.num_epochs
    EMBEDDING_DIM = args.embedding_dim
    LEARNING_RATE = args.learning_rate

    data = np.load(os.path.join('..','data',DATASET+'.npz'))

    triples,traces,entities,relations = utils.get_data(data,RULE)

    MAX_PADDING, LONGEST_TRACE = utils.get_longest_trace(DATASET, RULE)

    X_train_triples, X_train_traces, X_test_triples, X_test_traces = utils.train_test_split_no_unseen(
        triples,traces,longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,test_size=.25,seed=SEED)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = EMBEDDING_DIM
    THRESHOLD = .01

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    with strategy.scope():

        model = RGCN.get_RGCN_Model(
            num_entities=NUM_ENTITIES,
            num_relations=NUM_RELATIONS,
            embedding_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            seed=SEED
        )

        model.load_weights(os.path.join('..','data','weights',DATASET,DATASET+'_'+RULE+'.h5'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        init_value = tf.random.normal(
                (1,NUM_ENTITIES,NUM_ENTITIES), 
                mean=0, 
                stddev=1, 
                dtype=tf.float32, 
                seed=SEED
            )

        masks = [tf.Variable(
            initial_value=init_value,
            name='mask_'+str(i),
            trainable=True) for i in range(NUM_RELATIONS)
        ]

    train2idx = utils.array2idx(X_train_triples,ent2idx,rel2idx)
    trainexp2idx = utils.array2idx(X_train_traces,ent2idx,rel2idx)
    
    test2idx = utils.array2idx(X_test_triples,ent2idx,rel2idx)
    testexp2idx = utils.array2idx(X_test_traces,ent2idx,rel2idx)

    ADJACENCY_DATA = tf.concat([
        train2idx,
        trainexp2idx.reshape(-1,3),
        test2idx,
        testexp2idx.reshape(-1,3)
        ],axis=0
    )

    del train2idx
    del trainexp2idx

    TEST_SIZE = test2idx.shape[0]

    tf_data = tf.data.Dataset.from_tensor_slices(
        (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

    dist_dataset = strategy.experimental_distribute_dataset(tf_data)

    preds = []

    for head,rel,tail,explanation in dist_dataset:

        loss, current_preds = distributed_replica_step(head,rel,tail,explanation,NUM_ENTITIES,NUM_RELATIONS)

        preds.append(current_preds)

    best_preds = [array.numpy() for array in preds]

    out_preds = []

    for i in range(len(best_preds)):

        preds_i = utils.idx2array(best_preds[i],idx2ent,idx2rel)

        out_preds.append(preds_i)

    out_preds = np.array(out_preds,dtype=object)

    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f'learning_rate: {LEARNING_RATE}')
    print(f'threshold {THRESHOLD}')

    print(f"{DATASET} {RULE}")

    np.savez(os.path.join('..','data','preds',DATASET,'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),
        preds=best_preds
        )

    print('Done.')
