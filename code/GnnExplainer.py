#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN
from sklearn.metrics import jaccard_score
from scipy.sparse import csr_matrix

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

        non_masked_indices = tf.gather(mask_i.indices[mask_i.values > threshold], [1,2],axis=1)

        if non_masked_indices.shape[0]:

            pred_graph = csr_matrix(
                (tf.ones(non_masked_indices.shape[0]),(non_masked_indices[:,0],non_masked_indices[:,1])),
                shape=(num_entities,num_entities)
            )

            pred_graphs.append(non_masked_indices)

        else:

            pred_graph = csr_matrix(([],([],[])),shape=(num_entities,num_entities))
            pred_graphs.append([])

        true_indices = true_subgraphs[i].indices

        if true_indices.shape[0]:

            gather = tf.gather(true_indices,[1,2],axis=1)

            true_graph = csr_matrix(
                (true_subgraphs[i].values,(gather[:,0],gather[:,1])),
                shape=(num_entities,num_entities))

        else:
            true_graph = csr_matrix(([],([],[])),shape=(num_entities,num_entities))


        score = jaccard_score(true_graph,pred_graph,average='micro')

        scores.append(score)

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

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

    cv_scores = []
    cv_preds = []

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

    for train_idx,test_idx in kf.split(X=triples):

        train2idx = utils.array2idx(triples[train_idx],ent2idx,rel2idx)
        trainexp2idx = utils.array2idx(traces[train_idx],ent2idx,rel2idx)
        nopred2idx = utils.array2idx(nopred,ent2idx,rel2idx)
        
        #adjacency_data = tf.concat([train2idx,trainexp2idx.reshape(-1,3),nopred2idx],axis=0)

        test2idx = utils.array2idx(triples[test_idx],ent2idx,rel2idx)
        testexp2idx = utils.array2idx(traces[test_idx],ent2idx,rel2idx)

        adjacency_data = tf.concat([
            train2idx,
            trainexp2idx.reshape(-1,3),
            nopred2idx,
            test2idx,
            testexp2idx.reshape(-1,3)
            ],axis=0
        )

        del train2idx
        del trainexp2idx
        del nopred2idx

        jaccard_scores = []
        preds = []

        for i in range(test2idx.shape[0]):
        #for i in range(1):

            head = test2idx[i,0]
            rel = test2idx[i,1]
            tail = test2idx[i,2]

            comp_graph = get_computation_graph(head,rel,tail,K,adjacency_data,NUM_RELATIONS)

            adj_mats = utils.get_adj_mats(comp_graph, NUM_ENTITIES, NUM_RELATIONS)

            for epoch in range(NUM_EPOCHS):

                with tf.GradientTape(watch_accessed_variables=False) as tape:
                #with tf.GradientTape() as tape:

                    tape.watch(masks)

                    masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(NUM_RELATIONS)]

                    pred = model([
                        all_indices,
                        tf.reshape(head,(1,-1)),
                        tf.reshape(rel,(1,-1)),
                        tf.reshape(tail,(1,-1)),
                        masked_adjs
                        ]
                    )

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
            print(f"Jaccard {jaccard}")
            preds.append(pred_graphs)

        cv_avg = np.mean(jaccard_scores)

        print(f"CV Average: {cv_avg}")
        cv_scores.append(cv_avg)
        cv_preds.append(preds)

    best_idx = np.argmin(cv_scores)
    best_preds = cv_preds[best_idx]#.squeeze()

    print(f"Jaccard score: {np.mean(cv_scores)}")

    np.savez(os.path.join('..','data','preds','gnn_explainer_'+RULE+'_preds.npz'),
        preds=best_preds,embedding_dim=EMBEDDING_DIM,best_idx=best_idx,k=K,
        threshold=THRESHOLD,learning_rate=LEARNING_RATE,num_epochs=NUM_EPOCHS
        )
