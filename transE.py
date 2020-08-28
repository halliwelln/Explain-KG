#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
import numpy as np

def transE(num_entities,num_relations,embedding_size,random_state=123):

    '''Keras model for TransE'''

    initializer = 6/tf.sqrt(tf.cast(embedding_size, tf.float32))

    pos_head_input = Input(shape=(), name='pos_head_input')
    neg_head_input = Input(shape=(), name='neg_head_input')
    pos_tail_input = Input(shape=(), name='pos_tail_input')
    neg_tail_input = Input(shape=(), name='neg_tail_input')
    relation_input = Input(shape=(), name='relation_input')

    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_size,
        name='entity_embeddings',
        embeddings_initializer=RandomUniform(
            minval=-initializer,
            maxval=initializer,
            seed=random_state
            )
        )

    relation_embeddings = Embedding(
        input_dim=num_relations,
        output_dim=embedding_size,
        name='relation_embeddings',
        embeddings_initializer=RandomUniform(
            minval=-initializer,
            maxval=initializer,
            seed=random_state
            )
        )

    pos_head_e = entity_embeddings(pos_head_input)
    neg_head_e = entity_embeddings(neg_head_input)
    pos_tail_e = entity_embeddings(pos_tail_input)
    neg_tail_e = entity_embeddings(neg_tail_input)
    rel_e = relation_embeddings(relation_input)

    model = Model(
        inputs=[
            pos_head_input,
            pos_tail_input,
            neg_head_input, 
            neg_tail_input, 
            relation_input
            ], 
        outputs=[
            pos_head_e,
            pos_tail_e,
            neg_head_e,  
            neg_tail_e, 
            rel_e
            ]
        )

    return model

# def exp_loss(
#     pos_head_exp_e,
#     pos_tail_exp_e,
#     neg_head_exp_e, 
#     neg_tail_exp_e, 
#     rel_exp_e,
#     margin=2
#     ):
#     '''Explanation loss function'''

#     pos = tf.reduce_sum(tf.square(pos_head_exp_e + rel_exp_e - pos_tail_exp_e), axis=1)
#     neg = tf.reduce_sum(tf.square(neg_head_exp_e + rel_exp_e - neg_tail_exp_e), axis=1)    
            
#     loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))  

#     return loss

def exp_loss(
    pos_head_e,
    pos_tail_e,
    pos_head_exp_e,
    pos_tail_exp_e,
    rel_e, 
    rel_exp_e
    ):
    '''Explanation loss function'''

    squared_diff = tf.square(pos_head_e - pos_head_exp_e)\
     + tf.square(pos_tail_e - pos_tail_exp_e)+tf.square(rel_e - rel_exp_e)

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(squared_diff,axis=1)))

    return loss

def pred_loss(
    pos_head_e,
    pos_tail_e,
    neg_head_e,
    neg_tail_e,
    rel_e,
    margin=2
    ):
    '''Link prediction loss function'''

    pos = tf.reduce_sum(tf.square(pos_head_e + rel_e - pos_tail_e), axis=1)
    neg = tf.reduce_sum(tf.square(neg_head_e + rel_e - neg_tail_e), axis=1)    
            
    loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))  

    return loss

def link_score(head_e, rel_e, tail_e):
    return -np.sum(np.square(head_e + rel_e - tail_e))

def exp_score(triple,k,data,entity_embeddings,relation_embeddings):

    '''Get k closest L2 triples from <data>'''
    
    triple_h_e = entity_embeddings[triple[0]]
    triple_r_e = relation_embeddings[triple[1]]
    triple_t_e = entity_embeddings[triple[2]]
    
    h_e = entity_embeddings[data[:,0]]
    r_e = relation_embeddings[data[:,1]]
    t_e = entity_embeddings[data[:,2]]

    squared_diff = np.square(triple_h_e - h_e) + np.square(triple_r_e-r_e) + np.square(triple_t_e-t_e)

    l2_dist = np.sqrt(np.sum(squared_diff,axis=1))

    closest_l2 = np.argsort(l2_dist)[:k]
    
    return data[closest_l2]

def predict_link(head_e,tail_e,rel2idx,relations_str,relation_embeddings,score_fun,k=1):
    '''Predicts link between two nodes'''

    scores = []

    for rel in relations:

        rel_idx = rel2idx[rel]
        rel_e = relation_embeddings[rel_idx]

        score = score_fun(head_e,rel_e,tail_e)

        scores.append((rel,rel_idx,score))

    return sorted(scores,key=lambda x:x[2],reverse=True)[:k]

def predict_exp(triple,k):
    '''Predicts k explanations triples of an input triple'''
    pass
