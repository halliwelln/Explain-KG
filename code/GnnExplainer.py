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

    neighbors_head = tf.concat([subset[subset[:,0] == head],
                                subset[subset[:,2] == head]],axis=0)
    neighbors_tail = tf.concat([subset[(subset[:,0] == tail) & (subset[:,0] != head)],
                                subset[(subset[:,2] == tail) & (subset[:,0] != head)]],axis=0)

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

        sig_mask = tf.nn.sigmoid(masks)
        #loss = -1 * tf.math.log(y_pred + .00001) + tf.reduce_mean(tf.nn.sigmoid(masks))
        ent = - sig_mask * tf.math.log(sig_mask + .00001) - (1-sig_mask) * tf.math.log(1-sig_mask + .00001)
        loss = -1 * tf.math.log(y_pred + .00001) + tf.reduce_mean(ent)

    return loss, tape.gradient(loss,masks)

def score_subgraphs(
    true_subgraphs,
    adj_mats,
    masks,
    num_relations,
    num_entities,
    threshold
    ):
    
    '''Compute jaccard score across all relations for one triple'''

    scores = []

    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

        non_masked_indices = mask_i.indices[mask_i.values > threshold]

        pred_graph = tf.sparse.SparseTensor(
            indices=non_masked_indices,
            values=tf.ones(non_masked_indices.shape[0]),
            dense_shape=(1,num_entities,num_entities)
            )

        pred_graph = tf.sparse.to_dense(pred_graph)
        true_graph = tf.sparse.to_dense(true_subgraphs[i])
        print(true_graph.numpy().sum())
        print(pred_graph.numpy().sum())

        score = utils.tf_binary_jaccard(true_graph,pred_graph)

        if tf.math.is_nan(score):
            scores.append(0)
        else:
            scores.append(score)

    return tf.reduce_mean(scores[1:]) #ignore unknown relation

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import train_test_split

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('rule',type=str,help=
        'Enter which rule to use spouse,successor,...etc (str), -1 (str) for full dataset')
    args = parser.parse_args()

    RULE = args.rule

    #RULE = 'spouse'

    data = np.load(os.path.join('..','data','royalty.npz'))

    if RULE == '-1':
        triples,traces,no_pred_triples,no_pred_traces = utils.concat_triples(data, data['rules'])
        RULE = 'full_data'
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    else:
        triples, traces = data[RULE + '_triples'], data[RULE + '_traces']
        entities = data[RULE + '_entities'].tolist()
        relations = data[RULE + '_relations'].tolist()  

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 100
    LEARNING_RATE = .01
    NUM_EPOCHS = 20
    THRESHOLD = .5

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    # indices = []
    # for idx,i in enumerate(traces.reshape(-1,3)):
    #     if (i != ['UNK_ENT', 'UNK_REL', 'UNK_ENT']).all():
    #         indices.append(idx)

    # traces = traces.reshape(-1,3)[indices]
    # triples2idx = tf.convert_to_tensor(utils.array2idx(triples, ent2idx,rel2idx))
    # traces2idx = tf.convert_to_tensor(utils.array2idx(traces, ent2idx,rel2idx))

    triples2idx = utils.array2idx(triples, ent2idx,rel2idx)
    traces2idx = utils.array2idx(traces, ent2idx,rel2idx)

    # adj_mats = utils.get_adjacency_matrix_list(
    #     num_relations=NUM_RELATIONS,
    #     num_entities=NUM_ENTITIES,
    #     data=train2idx
    # )
    # triples2idx = tf.expand_dims(triples2idx,axis=0)
    # traces2idx = tf.expand_dims(traces2idx,axis=0)
    #train2idx = np.expand_dims(triples2idx,axis=0)

    all_indices = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))
    #all_indices = np.arange(NUM_ENTITIES).reshape(1,-1)

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.load_weights(os.path.join('..','data','weights',RULE+'.h5'))
    model.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    full_data = tf.concat([triples2idx,tf.reshape(traces2idx,(-1,3))],axis=0)

    relation_embeddings = model.get_layer('output').get_weights()[0]

    relation_kernel, self_kernel = model.get_layer('rgcn__layer').get_weights()

    entity_embeddings = model.get_layer('entity_embeddings').get_weights()[0]

    scores = []
    for i in range(1):
        # head = tf.reshape(triples2idx[i,0],shape=(-1,1))
        # rel = tf.reshape(triples2idx[i,1],shape=(-1,1))
        # tail = tf.reshape(triples2idx[i,2],shape=(-1,1))

        head = triples2idx[i,0]
        rel = triples2idx[i,1]
        tail = triples2idx[i,2]

        true_subgraphs = utils.get_adj_mats(traces2idx[i],NUM_ENTITIES,NUM_RELATIONS)

        comp_graph = get_computation_graph(head,rel,tail,full_data,NUM_RELATIONS)

        adj_mats = utils.get_adj_mats(comp_graph, NUM_ENTITIES, NUM_RELATIONS)

        masks = [tf.Variable(
                initial_value=tf.random.normal(
                    (1,NUM_ENTITIES,NUM_ENTITIES), 
                    mean=0, 
                    stddev=1, 
                    dtype=tf.float32, 
                    seed=SEED),
                name='mask_'+str(i),
                trainable=True) for i in range(NUM_RELATIONS)
        ]
        # masks = [tf.Variable(
        #         initial_value=tf.sparse.to_dense(adj_mats[i]),
        #         name='mask_'+str(i),
        #         trainable=
            #     adj_mats,
            #     model,True) for i in range(NUM_RELATIONS)
        #]
        # masks = [tf.Variable(
        #         initial_value=tf.ones((NUM_ENTITIES,NUM_ENTITIES)),
        #         name='mask_'+str(i),
        #         trainable=True) for i in range(NUM_RELATIONS)
        # ]

        for epoch in range(NUM_EPOCHS):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
            #with tf.GradientTape() as tape:

                tape.watch(masks)

                head_output = tf.matmul(tf.reshape(entity_embeddings[head],(1,-1)),self_kernel)
                tail_output = tf.matmul(tf.reshape(entity_embeddings[tail],(1,-1)),self_kernel)

                for i in range(NUM_RELATIONS):
                    adj_i = masks[i][0]
                    sum_embeddings = tf.matmul(adj_i,entity_embeddings)
                    head_update = tf.reshape(sum_embeddings[head],(1,-1))
                    tail_update = tf.reshape(sum_embeddings[tail],(1,-1))
                    
                    head_output += tf.matmul(head_update,relation_kernel[i])
                    tail_output += tf.matmul(tail_update,relation_kernel[i])
                
                head_output = tf.sigmoid(head_output)
                tail_output = tf.sigmoid(tail_output)

                score = tf.matmul(tf.matmul(head_output,
                    relation_kernel[rel]),tf.transpose(tail_output))

                loss = -1 * tf.math.log(tf.sigmoid(score))

                # masked_adjs = []

                # for i in range(NUM_RELATIONS):

                #     mask_adj = adj_mats[i] * tf.nn.sigmoid(masks[i])

                #     thresholded_indices = mask_adj.indices[mask_adj.values > THRESHOLD]
    
                #     masked_adj = tf.sparse.SparseTensor(
                #             indices=thresholded_indices,
                #             values=tf.ones(thresholded_indices.shape[0]),
                #             dense_shape=(1,NUM_ENTITIES,NUM_ENTITIES)
                #             )
    
                #     masked_adjs.append(masked_adj)

                # y_pred = model(
                #     [
                #     all_indices,
                #     tf.reshape(head,shape=(-1,1)),
                #     tf.reshape(rel,shape=(-1,1)),
                #     tf.reshape(tail,shape=(-1,1)),
                #     masked_adjs
                #     ]
                # )

                # sig_masks = tf.nn.sigmoid(masks)

                # penalty = tf.reduce_sum(sig_masks)

                # loss = -1 * tf.math.log(y_pred +.00001) #+ (0.000001*penalty)

            print(f"Loss {tf.squeeze(loss).numpy()} @ epoch {epoch}")
            grads = tape.gradient(loss,masks)

            optimizer.apply_gradients(zip(grads,masks))

        score = score_subgraphs(
            true_subgraphs=true_subgraphs,
            adj_mats=adj_mats,
            masks=masks,
            num_relations=NUM_RELATIONS,
            num_entities=NUM_ENTITIES,
            threshold=THRESHOLD
        )
        scores.append(score)

        print(f"score {score.numpy()}")
    print(np.mean(scores))

