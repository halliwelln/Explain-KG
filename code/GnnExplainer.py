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

def get_computation_graph(head,rel,tail,k,data,num_relations):

    '''Get k hop neighbors (may include duplicates)'''

    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    # if k > 1:
    #     num_indices = all_neighbors.shape[0]

    #     seen_nodes = []
        
    #     for _ in range(k-1):#-1 since we already computed 1st degree neighbors above

    #         for idx in range(num_indices):

    #             head_neighbor_idx = all_neighbors[idx,0]
    #             tail_neighbor_idx = all_neighbors[idx,2]

    #             if head_neighbor_idx not in seen_nodes:
                    
    #                 seen_nodes.append(head_neighbor_idx)

    #                 more_head_neighbors = get_neighbors(data,head_neighbor_idx)

    #                 all_neighbors = tf.concat([all_neighbors,more_head_neighbors],axis=0)

    #             if tail_neighbor_idx not in seen_nodes:

    #                 seen_nodes.append(tail_neighbor_idx)

    #                 more_tail_neighbors = get_neighbors(data,tail_neighbor_idx)

    #                 all_neighbors = tf.concat([all_neighbors,more_tail_neighbors],axis=0)

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

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import KFold
    import tensorflow as tf

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

    def replica_step(head,rel,tail,explanation):
        
        comp_graph = get_computation_graph(head,rel,tail,K,ADJACENCY_DATA,NUM_RELATIONS)

        adj_mats = utils.get_adj_mats(comp_graph, NUM_ENTITIES, NUM_RELATIONS)

        true_subgraphs = utils.get_adj_mats(tf.squeeze(explanation,axis=0),NUM_ENTITIES,NUM_RELATIONS)

        total_loss = 0.0

        for epoch in range(NUM_EPOCHS):

            with tf.GradientTape(watch_accessed_variables=False) as tape:

                tape.watch(masks)

                masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(NUM_RELATIONS)]

                pred = model([
                        ALL_INDICES,
                        tf.reshape(head,(1,-1)),
                        tf.reshape(rel,(1,-1)),
                        tf.reshape(tail,(1,-1)),
                        masked_adjs
                        ]
                    )

                penalty = [tf.reduce_sum(tf.cast(tf.sigmoid(i.values) > .5,dtype=tf.float32)) for i in masked_adjs]

                loss = -1 * tf.math.log(pred+0.00001) + (0.0001 * tf.reduce_sum(penalty))

                tf.print(f"current loss {loss}")

                total_loss += loss

            grads = tape.gradient(loss,masks)
            optimizer.apply_gradients(zip(grads,masks))

        current_preds = []
        total_jaccard = 0.0

        for i in range(NUM_RELATIONS):

            mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

            non_masked_indices = mask_i.indices[mask_i.values > THRESHOLD]

            if (non_masked_indices.shape[0] == 0) and (tf.math.reduce_all(true_subgraphs[i].values == tf.zeros((1,3)))):
                total_jaccard += 1.
            else:
                total_jaccard += tf_jaccard(true_subgraphs[i].indices,non_masked_indices)

            pred = non_masked_indices.numpy()

            current_preds.append(pred[:,1:])

        total_jaccard /= NUM_RELATIONS

        #tf.print(f"per observation jaccard: {total_jaccard}")

        for mask in masks:
            mask.assign(value=init_value)

        return total_loss, total_jaccard, current_preds

    def distributed_replica_step(head,rel,tail,explanation):

        per_replica_losses, per_replica_jaccard, current_preds = strategy.run(replica_step, args=(head,rel,tail,explanation))

        reduce_loss = per_replica_losses / NUM_EPOCHS

        #tf.print(f"reduced loss {reduce_loss}")
        #tf.print(f"reduced jaccard {per_replica_jaccard}")

        return reduce_loss,per_replica_jaccard, current_preds

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
    LEARNING_RATE = .001
    NUM_EPOCHS = 10
    THRESHOLD = .01
    K = 1

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

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

        model.load_weights(os.path.join('..','data','weights',RULE+'.h5'))

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

    cv_scores = []
    cv_preds = []
    test_indices = []

    for train_idx,test_idx in kf.split(X=triples):

        #test_idx = test_idx[0:3]

        preds = []

        train2idx = utils.array2idx(triples[train_idx],ent2idx,rel2idx)
        trainexp2idx = utils.array2idx(traces[train_idx],ent2idx,rel2idx)
        nopred2idx = utils.array2idx(nopred,ent2idx,rel2idx)
        
        test2idx = utils.array2idx(triples[test_idx],ent2idx,rel2idx)
        testexp2idx = utils.array2idx(traces[test_idx],ent2idx,rel2idx)

        ADJACENCY_DATA = tf.concat([
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

        TEST_SIZE = test2idx.shape[0]

        tf_data = tf.data.Dataset.from_tensor_slices(
            (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

        dist_dataset = strategy.experimental_distribute_dataset(tf_data)

        total_jaccard = 0.0

        for head,rel,tail,explanation in dist_dataset:

            loss, jaccard, current_preds = distributed_replica_step(head,rel,tail,explanation)
    
            preds.append(current_preds)

            total_jaccard += jaccard

        total_jaccard /= TEST_SIZE

        print(f"CV jaccard: {total_jaccard}")

        cv_scores.append(total_jaccard)
        cv_preds.append(preds)
        test_indices.append(test_idx)

    best_idx = np.argmax(cv_scores)
    best_preds = np.array(cv_preds[best_idx])
    best_test_indices = test_indices[best_idx]

    print(f"{RULE} jaccard score: {cv_scores[best_idx]}")
    print(f"using learning rate: {LEARNING_RATE}, and {NUM_EPOCHS} epochs")
    print(f"threshold {THRESHOLD}, and k={K}")

    np.savez(os.path.join('..','data','preds','gnn_explainer_'+RULE+'_preds.npz'),
        best_idx=best_idx, preds=best_preds,test_idx=best_test_indices
        )

    print('Done.')
