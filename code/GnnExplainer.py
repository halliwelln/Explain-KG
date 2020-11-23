#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN

def get_neighbors(data_subset,node_idx):
    
    neighbors = tf.concat([data_subset[data_subset[:,0] == node_idx],
                           data_subset[data_subset[:,2] == node_idx]],axis=0)
    
    return neighbors

def get_computation_graph(head,rel,tail,k,data,num_relations):

    '''Get k hop neighbors (may include duplicates)'''
         
    # subset = data[data[:,1] == rel]

    # neighbors_head = get_neighbors(subset,head)
    # neighbors_tail = get_neighbors(subset,tail)

    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    if k > 1:
        num_indices = all_neighbors.shape[0]

        seen_nodes = []
        
        for _ in range(k-1):#-1 since we already computed 1st degree neighbors above

            for idx in range(num_indices):

                head_neighbor_idx = all_neighbors[idx,0]
                tail_neighbor_idx = all_neighbors[idx,2]

                if head_neighbor_idx not in seen_nodes:
                    
                    seen_nodes.append(head_neighbor_idx)

                    more_head_neighbors = get_neighbors(data,head_neighbor_idx)

                    all_neighbors = tf.concat([all_neighbors,more_head_neighbors],axis=0)

                if tail_neighbor_idx not in seen_nodes:

                    seen_nodes.append(tail_neighbor_idx)

                    more_tail_neighbors = get_neighbors(data,tail_neighbor_idx)

                    all_neighbors = tf.concat([all_neighbors,more_tail_neighbors],axis=0)

    return all_neighbors

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
    pred_graphs = []

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

        #print(np.argwhere(pred_graph.numpy()))
        #print(np.argwhere(true_graph.numpy()))

        score = utils.tf_binary_jaccard(true_graph,pred_graph)

        if tf.math.is_nan(score):
            scores.append(0)
        else:
            scores.append(score)

        graph = np.argwhere(pred_graph.numpy().squeeze())
        col = np.ones((graph.shape[0],1),dtype=np.int64) * i
        out_graph = np.concatenate([graph[:,0].reshape(-1,1),col,graph[:,1].reshape(-1,1)],axis=1)

        if out_graph.shape[0]:
            pred_graphs.append(out_graph)
        else:
            pred_graphs.append([])

    return tf.reduce_mean(scores[1:]), pred_graphs#ignore unknown relation

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import KFold

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('rule',type=str,help=
        'Enter which rule to use spouse,successor,...etc (str), full_data for full dataset')
    args = parser.parse_args()

    RULE = args.rule

    data = np.load(os.path.join('..','data','royalty.npz'))

    if RULE == 'full_data':
        triples,traces,nopred = utils.concat_triples(data, data['rules'])
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    else:
        triples,traces,nopred = utils.concat_triples(data, [RULE,'brother','sister'])
        sister_relations = data['sister_relations'].tolist()
        sister_entities = data['sister_entities'].tolist()

        brother_relations = data['brother_relations'].tolist()
        brother_entities = data['brother_entities'].tolist()

        entities = np.unique(data[RULE + '_entities'].tolist()+brother_entities+sister_entities).tolist()
        relations = np.unique(data[RULE + '_relations'].tolist()+brother_relations+sister_relations).tolist()

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    EMBEDDING_DIM = 50
    OUTPUT_DIM = 50
    LEARNING_RATE = .01
    NUM_EPOCHS = 1
    THRESHOLD = .01
    K = 1

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    #triples2idx = utils.array2idx(triples, ent2idx,rel2idx)
    #traces2idx = utils.array2idx(traces, ent2idx,rel2idx)

    all_indices = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.load_weights(os.path.join('..','data','weights',RULE+'.h5'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    relation_embeddings = model.get_layer('output').get_weights()[0]

    relation_kernel, self_kernel = model.get_layer('rgcn__layer').get_weights()

    entity_embeddings = model.get_layer('entity_embeddings').get_weights()[0]

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

    cv_scores = []
    cv_preds = []

    for train_idx,test_idx in kf.split(X=triples):

        #test_idx = test_idx[0:2]

        train2idx = utils.array2idx(triples[train_idx],ent2idx,rel2idx)
        trainexp2idx = utils.array2idx(traces[train_idx],ent2idx,rel2idx)
        nopred2idx = utils.array2idx(nopred,ent2idx,rel2idx)
        
        adjacency_data = tf.concat([train2idx,trainexp2idx.reshape(-1,3),nopred2idx],axis=0)

        test2idx = utils.array2idx(triples[test_idx],ent2idx,rel2idx)
        testexp2idx = utils.array2idx(traces[test_idx],ent2idx,rel2idx)

        jaccard_scores = []
        preds = []
        
        for i in range(test2idx.shape[0]):

            head = test2idx[i,0]
            rel = test2idx[i,1]
            tail = test2idx[i,2]

            comp_graph = get_computation_graph(head,rel,tail,K,adjacency_data,NUM_RELATIONS)

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

            for epoch in range(NUM_EPOCHS):

                with tf.GradientTape(watch_accessed_variables=False) as tape:
                #with tf.GradientTape() as tape:

                    tape.watch(masks)

                    head_output = tf.matmul(tf.reshape(entity_embeddings[head],(1,-1)),self_kernel)
                    tail_output = tf.matmul(tf.reshape(entity_embeddings[tail],(1,-1)),self_kernel)

                    for i in range(NUM_RELATIONS):

                        adj_i = tf.sparse.to_dense(adj_mats[i])[0] * tf.sigmoid(masks[i][0])

                        sum_embeddings = tf.matmul(adj_i,entity_embeddings)

                        head_update = tf.reshape(sum_embeddings[head],(1,-1))
                        tail_update = tf.reshape(sum_embeddings[tail],(1,-1))
                        
                        head_output += tf.matmul(head_update,relation_kernel[i])
                        tail_output += tf.matmul(tail_update,relation_kernel[i])
                    
                    head_output = tf.sigmoid(head_output)
                    tail_output = tf.sigmoid(tail_output)

                    pred = tf.sigmoid(tf.reduce_sum(head_output*relation_kernel[rel]*tail_output))

                    loss = -1 * tf.math.log(pred+0.00001)# + tf.reduce_mean(masks)

                print(f"Loss {tf.squeeze(loss).numpy()} @ epoch {epoch}")

                grads = tape.gradient(loss,masks)
                optimizer.apply_gradients(zip(grads,masks))

            true_subgraphs = utils.get_adj_mats(testexp2idx[i],NUM_ENTITIES,NUM_RELATIONS)

            jaccard, pred_graphs = score_subgraphs(
                true_subgraphs=true_subgraphs,
                adj_mats=adj_mats,
                masks=masks,
                num_relations=NUM_RELATIONS,
                num_entities=NUM_ENTITIES,
                threshold=THRESHOLD
            )
            jaccard_scores.append(jaccard)
            preds.append(pred_graphs)

        cv_scores.append(np.mean(jaccard_scores))
        cv_preds.append(preds)

    best_idx = np.argmin(cv_scores)
    best_preds = cv_preds[best_idx]#.squeeze()

    print(f"Jaccard score: {np.mean(cv_scores)}")

    np.savez(os.path.join('..','data','preds','gnn_explainer_'+RULE+'_preds.npz'),
        preds=best_preds,embedding_dim=EMBEDDING_DIM,best_idx=best_idx,k=K,
        threshold=THRESHOLD,learning_rate=LEARNING_RATE,num_epochs=NUM_EPOCHS
        )


