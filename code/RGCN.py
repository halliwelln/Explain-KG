#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import utils

class RGCN_Layer(tf.keras.layers.Layer):
    def __init__(self,num_entities,num_relations,output_dim,seed,**kwargs):
        super(RGCN_Layer,self).__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.seed = seed
        
    def build(self,input_shape):

        input_dim = int(input_shape[-2][-1])
        
        self.relation_kernel = self.add_weight(
            shape=(self.num_relations,input_dim, self.output_dim),
            name="relation_kernels",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            name="self_kernel",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )
    
    # def call(self, inputs):
        
    #     embeddings,head_idx,head_e,tail_idx,tail_e,adj_mats = inputs
            
    #     head_output = tf.matmul(head_e,self.self_kernel)
    #     tail_output = tf.matmul(tail_e,self.self_kernel)
                
    #     for i in range(self.num_relations):
            
    #         adj_i = adj_mats[i]
            
    #         head_adj = tf.nn.embedding_lookup(adj_i,head_idx)
    #         tail_adj = tf.nn.embedding_lookup(adj_i,tail_idx)
            
    #         head_update = tf.matmul(head_adj,embeddings)
    #         tail_update = tf.matmul(tail_adj,embeddings)

    #         head_output += tf.matmul(head_update,self.relation_kernel[i])
    #         tail_output += tf.matmul(tail_update,self.relation_kernel[i])
       
    #     return head_output, tail_output

    def call(self,inputs):

        embeddings,head_idx,head_e,tail_idx,tail_e,*adj_mats = inputs
                        
        head_output = tf.matmul(head_e,self.self_kernel)
        tail_output = tf.matmul(tail_e,self.self_kernel)
                
        for i in range(self.num_relations):
            
            adj_i = tf.sparse.reshape(adj_mats[0][i],shape=(self.num_entities,self.num_entities))

            sum_embeddings = tf.sparse.sparse_dense_matmul(adj_i, embeddings)
            
            head_update = tf.nn.embedding_lookup(sum_embeddings,head_idx)
            tail_update = tf.nn.embedding_lookup(sum_embeddings,tail_idx)
            
            head_output += tf.matmul(head_update,self.relation_kernel[i])
            tail_output += tf.matmul(tail_update,self.relation_kernel[i])
       
        return tf.sigmoid(head_output), tf.sigmoid(tail_output)

class DistMult(tf.keras.layers.Layer):
    def __init__(self, num_relations,seed,**kwargs):
        super(DistMult,self).__init__(**kwargs)
        self.num_relations = num_relations
        self.seed = seed
        
    def build(self,input_shape):
        
        embedding_dim = input_shape[0][-1]
        
        self.kernel = self.add_weight(
            shape=(self.num_relations,embedding_dim),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            ),
            name='rel_embedding'
        )
        
    def call(self,inputs):
        
        head_e,rel_idx,tail_e = inputs
        
        rel_e = tf.nn.embedding_lookup(self.kernel,rel_idx)

        score = tf.sigmoid(tf.reduce_sum(head_e*rel_e*tail_e,axis=-1))

        return tf.expand_dims(score,axis=0)

class RGCN_Model(tf.keras.Model):

    def __init__(self,num_entities,*args,**kwargs):
        super(RGCN_Model,self).__init__(*args, **kwargs)
        self.num_entities = num_entities

    def train_step(self,data):

        all_indices,pos_head,rel,pos_tail,*adj_mats = data[0]
        y_pos_true = data[1]

        neg_head, neg_tail = utils.get_negative_triples(
                head=pos_head, 
                rel=rel, 
                tail=pos_tail,
                num_entities=self.num_entities
            )

        with tf.GradientTape() as tape:

            y_pos_pred = self([
                    all_indices,
                    pos_head,
                    rel,
                    pos_tail,
                    adj_mats
                    ],
                    training=True
                )

            y_neg_pred = self([
                    all_indices,
                    neg_head,
                    rel,
                    neg_tail,
                    adj_mats
                    ],
                    training=True
                )

            y_pred = tf.concat([y_pos_pred,y_neg_pred],axis=1)
            y_true = tf.concat([y_pos_true,tf.zeros_like(y_pos_true)],axis=1)

            loss = self.compiled_loss(y_true,y_pred)

            loss *= (1/ self.num_entities)
            
            # pos_loss = self.compiled_loss(y_pos_true,y_pos_pred)
            # neg_loss = self.compiled_loss(y_neg_true,y_neg_pred)

            # loss = (pos_loss + neg_loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.compiled_metrics.update_state(y_pos_true, y_pos_pred)

        return {m.name: m.result() for m in self.metrics}

def get_RGCN_Model(num_entities,num_relations,embedding_dim,output_dim,seed):

    head_input = tf.keras.Input(shape=(None,), name='head_input',dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(None,), name='rel_input',dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(None,), name='tail_input',dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(None,), name='all_entities',dtype=tf.int64)

    adj_inputs = [tf.keras.Input(
        shape=(num_entities,num_entities),
        dtype=tf.float32,
        name='adj_inputs_'+str(i),
        sparse=True,
        ) for i in range(num_relations)]

    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_dim,
        name='entity_embeddings',
        embeddings_initializer=tf.keras.initializers.RandomUniform(
            minval=-1,
            maxval=1,
            seed=seed
        )
    )

    head_e = entity_embeddings(head_input)
    tail_e = entity_embeddings(tail_input)
    all_e = entity_embeddings(all_entities)

    head_e = Lambda(lambda x:x[0,:,:])(head_e)
    tail_e = Lambda(lambda x:x[0,:,:])(tail_e)
    all_e = Lambda(lambda x:x[0,:,:])(all_e)

    head_index = Lambda(lambda x:x[0,:])(head_input)
    rel_index = Lambda(lambda x:x[0,:])(rel_input)
    tail_index = Lambda(lambda x:x[0,:])(tail_input)

    new_head,new_tail = RGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
            all_e,
            head_index,
            head_e,
            tail_index,
            tail_e,
            adj_inputs
            ]
        )

    output = DistMult(num_relations=num_relations,seed=seed,name='output')([
        new_head,rel_index,new_tail
        ]
    )

    model = RGCN_Model(
        inputs=[all_entities,head_input,rel_input,tail_input] + adj_inputs,
        outputs=[output],
        num_entities=num_entities
    )

    return model

if __name__ == '__main__':

    import numpy as np
    import argparse
    import os
    import utils
    import random as rn
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
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 100    
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    triples2idx = utils.array2idx(triples,ent2idx,rel2idx)
    traces2idx = utils.array2idx(traces,ent2idx,rel2idx)

    #X_train = np.concatenate([triples2idx,traces2idx.reshape(-1,3)])
    # X_train,X_test,y_train,y_test = train_test_split(
    #     triples2idx,
    #     traces2idx,
    #     test_size=0.3,
    #     random_state=SEED
    # )

    # X_train = np.concatenate([X_train,y_train.reshape(-1,3)],axis=0)

    # if RULE == 'full_data':
    #     no_pred_triples2idx = utils.array2idx(no_pred_triples,ent2idx,rel2idx)
    #     no_pred_traces2idx = utils.array2idx(no_pred_traces,ent2idx,rel2idx).reshape(-1,3)
    #     X_train = np.concatenate([X_train,no_pred_triples,no_pred_traces],axis=0)

    # X_train = np.unique(X_train,axis=0)

    full_data = np.concatenate([triples2idx,traces2idx.reshape(-1,3)],axis=0)

    idx_train,idx_test = utils.train_test_split_no_unseen(
        full_data, 
        test_size=1500,
        seed=SEED, 
        allow_duplication=False, 
        filtered_test_predicates=None)

    X_train = full_data[idx_train]
    X_test = full_data[idx_test]

    NUM_TRIPLES = X_train.shape[0]

    adj_mats = utils.get_adj_mats(X_train,NUM_ENTITIES,NUM_RELATIONS)

    X_train = np.expand_dims(X_train,axis=0)
    X_test = np.expand_dims(X_test,axis=0)

    all_indices = np.arange(NUM_ENTITIES).reshape(1,-1)
    
    # strategy = tf.distribute.MirroredStrategy()
    # print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # with strategy.scope():
    model = get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )

    model.fit(
        x=[
            all_indices,
            X_train[:,:,0],
            X_train[:,:,1],
            X_train[:,:,2],
            adj_mats
            ],
        y=np.ones(NUM_TRIPLES).reshape(1,-1),
        epochs=NUM_EPOCHS,
        batch_size=1,
        verbose=1
    )

    #model.save_weights(os.path.join('..','data','weights',RULE+'.h5'))

    preds = model.predict(
        x=[
            all_indices,
            X_test[:,:,0],
            X_test[:,:,1],
            X_test[:,:,2],
            adj_mats
        ]
    )
    print(f'acc {(preds > .5).sum()/X_test.shape[1]}')
