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
       
        return head_output, tail_output


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

            y_neg_true = tf.zeros_like(y_pos_true)

            # y_pred = tf.concat([y_pos_pred,y_neg_pred],axis=1)
            # y_true = tf.concat([y,tf.zeros_like(y)],axis=1)
            
            pos_loss = self.compiled_loss(y_pos_true,y_pos_pred)
            neg_loss = self.compiled_loss(y_neg_true,y_neg_pred)

            loss = pos_loss + neg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.compiled_metrics.update_state(y_pos_true, y_pos_pred)

        return {m.name: m.result() for m in self.metrics}

def get_RGCN_Model(num_triples,num_entities,num_relations,embedding_dim,output_dim,seed):

    head_input = tf.keras.Input(shape=(num_triples,), name='head_input',dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(num_triples,), name='rel_input',dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(num_triples,), name='tail_input',dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(num_entities,), name='all_entities',dtype=tf.int64)

    # adj_inputs = tf.keras.Input(
    #     shape=(
    #         num_relations,
    #         num_entities,
    #         num_entities
    #     ),
    #     dtype=tf.float32,
    #     name='adj_inputs'
    # )
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

    #adj_mats_layers = [Lambda(lambda x:x[0,:,:])(adj_inputs[i]) for i in range(num_relations)]

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

    #output = tf.keras.layers.Dense(num_triples,activation='sigmoid')(output)

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
    EMBEDDING_DIM = 25
    OUTPUT_DIM = 50
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 2000

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    triples, traces = data['spouse_triples'], data['spouse_traces']

    train2idx = utils.array2idx(triples,ent2idx,rel2idx)

    NUM_TRIPLES = train2idx.shape[0]

    # adj_mats = utils.get_adjacency_matrix_list(
    #     num_relations=NUM_RELATIONS,
    #     num_entities=NUM_ENTITIES,
    #     data=train2idx
    # )
    def get_adj_mats(data,num_entities,num_relations):

        adj_mats = []

        for i in range(num_relations):

            data_i = data[data[:,1] == i]

            indices = np.concatenate([data_i[:,[0,2]],data_i[:,[2,0]]],axis=0)

            sparse_mat = tf.sparse.SparseTensor(
                indices=indices,
                values=np.ones((indices.shape[0])),
                dense_shape=(num_entities,num_entities)
                )

            sparse_mat = tf.sparse.reorder(sparse_mat)

            sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1,num_entities,num_entities))

            adj_mats.append(sparse_mat)

        return adj_mats

    adj_mats = get_adj_mats(train2idx,NUM_ENTITIES,NUM_RELATIONS)

    train2idx = np.expand_dims(train2idx,axis=0)

    all_indices = np.arange(NUM_ENTITIES).reshape(1,-1)
    
    # strategy = tf.distribute.MirroredStrategy()
    # print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # with strategy.scope():
    model = get_RGCN_Model(
        num_triples=NUM_TRIPLES,
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), 
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    )

    model.fit(
        x=[
            all_indices,
            train2idx[:,:,0],
            train2idx[:,:,1],
            train2idx[:,:,2],
            adj_mats
            ],
        y=np.ones(NUM_TRIPLES).reshape(1,-1),
        epochs=NUM_EPOCHS,
        batch_size=1,
        verbose=1
    )

    model.save_weights(os.path.join('..','data','weights','rgcn.h5'))

    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # bce = tf.keras.losses.BinaryCrossentropy()

    # data = tf.data.Dataset.from_tensor_slices((
    #         train2idx[:,:,0],
    #         train2idx[:,:,1],
    #         train2idx[:,:,2], 
    #         np.ones(train2idx.shape[1]),reshape(1,-1)
    #     )
    # ).batch(1)

    # for epoch in range(NUM_EPOCHS):

    #     for pos_head,rel,pos_tail,y in data:

    #         neg_head, neg_tail = utils.get_negative_triples(
    #             head=pos_head, 
    #             rel=rel, 
    #             tail=pos_tail,
    #             num_entities=NUM_ENTITIES
    #         )

    #         with tf.GradientTape() as tape:

    #             y_pos_pred = model([
    #                 all_indices,
    #                 pos_head,
    #                 rel,
    #                 pos_tail,
    #                 adj_mats
    #                 ],
    #                 training=True
    #             )
            
    #             y_neg_pred = model([
    #                 all_indices,
    #                 neg_head,
    #                 rel,
    #                 neg_tail,
    #                 adj_mats
    #                 ],
    #                 training=True
    #             )

    #             y_pred = tf.concat([y_pos_pred,y_neg_pred],axis=1)
    #             y_true = tf.concat([y,tf.zeros_like(y)],axis=1)
                
    #             loss = bce(y_true,y_pred)

    #         grads = tape.gradient(loss, model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #     print(f'loss {loss} after epoch {epoch}')

    preds = model.predict(
        x=[
            all_indices,
            train2idx[:,:,0],
            train2idx[:,:,1],
            train2idx[:,:,2],
            adj_mats
        ]
    )
    print(f'acc {(preds > .5).sum()/NUM_TRIPLES}')
