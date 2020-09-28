#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.initializers import RandomUniform
import numpy as np
import utils

def ExTransE(num_entities,num_relations,embedding_size,random_state=123):

    initializer = 6/tf.sqrt(tf.cast(embedding_size, tf.float32))

    head_input = Input(shape=(), name='head_input')
    rel_input = Input(shape=(), name='rel_input')
    tail_input = Input(shape=(), name='tail_input')

    exp_head_input = Input(shape=(), name='exp_head_input')
    exp_rel_input = Input(shape=(), name='exp_rel_input')
    exp_tail_input = Input(shape=(), name='exp_tail_input')

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

    head_e = entity_embeddings(head_input)
    rel_e = relation_embeddings(rel_input)
    tail_e = entity_embeddings(tail_input)

    exp_head_e = entity_embeddings(exp_head_input)
    exp_rel_e = relation_embeddings(exp_rel_input)
    exp_tail_e = entity_embeddings(exp_tail_input)

    model = ExTransE_Model(
        inputs=[
            head_input,
            rel_input,
            tail_input,
            exp_head_input,
            exp_rel_input,
            exp_tail_input
            ], 
        outputs=[
            head_e,
            rel_e,
            tail_e,
            exp_head_e,
            exp_rel_e,
            exp_tail_e
            ],
        num_entities=num_entities
        )

    return model

class ExTransE_Model(tf.keras.Model):

    def __init__(self,num_entities,*args,**kwargs):
        super(ExTransE_Model,self).__init__(*args, **kwargs)
        self.num_entities = num_entities

    def compile(self,optimizer,margin,pred_loss,exp_loss):
        super(ExTransE_Model,self).compile()
        self.optimizer = optimizer
        self.margin = margin
        self.pred_loss = pred_loss
        self.exp_loss = exp_loss

    def train_step(self,data):

        pos_head, pos_rel, pos_tail, pos_head_exp,pos_rel_exp, pos_tail_exp = data[0]

        neg_head, neg_tail = utils.get_negative_triples(
            head=pos_head, 
            rel=pos_rel, 
            tail=pos_tail,
            num_entities=self.num_entities
            )

        neg_head_exp, neg_tail_exp = utils.get_negative_triples(
            head=pos_head_exp, 
            rel=pos_rel_exp, 
            tail=pos_tail_exp,
            num_entities=self.num_entities
            )

        with tf.GradientTape() as tape:

            pos_head_e,pos_rel_e,pos_tail_e,pos_head_exp_e,pos_rel_exp_e,pos_tail_exp_e = self([
                pos_head,
                pos_rel,
                pos_tail,
                pos_head_exp,
                pos_rel_exp,
                pos_tail_exp
                ]
            )

            neg_head_e,neg_rel_e,neg_tail_e,neg_head_exp_e,neg_rel_exp_e,neg_tail_exp_e = self([
                neg_head,
                pos_rel,#pos_rel is correct, 
                neg_tail,
                neg_head_exp,
                pos_rel_exp,
                neg_tail_exp
                ]
            )

            prediction_loss = self.pred_loss(
                pos_head_e,
                pos_rel_e,
                pos_tail_e,
                neg_head_e,
                neg_rel_e,
                neg_tail_e,
                margin=self.margin
            )

            # explain_loss = self.exp_loss(
            #     pos_head_exp_e,
            #     pos_rel_exp_e,
            #     pos_tail_exp_e,
            #     neg_head_exp_e,
            #     neg_rel_exp_e,
            #     neg_tail_exp_e,
            #     margin=self.margin
            # )

            explain_loss = self.exp_loss(
                pos_head_e,
                pos_rel_e, 
                pos_tail_e,
                pos_head_exp_e,
                pos_rel_exp_e,
                pos_tail_exp_e
                )

            total_loss = prediction_loss + explain_loss

        grads = tape.gradient(total_loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {"loss":total_loss}

# def exp_loss(
#     pos_head_exp_e,
#     pos_rel_exp_e,
#     pos_tail_exp_e,
#     neg_head_exp_e, 
#     neg_rel_exp_e,
#     neg_tail_exp_e, 
#     margin=2
#     ):
#     '''Explanation loss function'''

#     pos = tf.reduce_sum(tf.square(pos_head_exp_e + pos_rel_exp_e - pos_tail_exp_e), axis=1)
#     neg = tf.reduce_sum(tf.square(neg_head_exp_e + pos_rel_exp_e - neg_tail_exp_e), axis=1)    
            
#     loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))  

#     return loss

def exp_loss(
    pos_head_e,
    pos_rel_e, 
    pos_tail_e,
    pos_head_exp_e,
    pos_rel_exp_e,
    pos_tail_exp_e
    ):
    '''Explanation loss function'''

    squared_diff = tf.square(pos_head_e - pos_head_exp_e)\
     + tf.square(pos_tail_e - pos_tail_exp_e)+tf.square(pos_rel_e - pos_rel_exp_e)

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(squared_diff,axis=1)))

    return loss

def pred_loss(
    pos_head_e,
    pos_rel_e,
    pos_tail_e,
    neg_head_e,
    neg_rel_e,
    neg_tail_e,
    margin=2
    ):
    '''Link prediction loss function'''

    pos = tf.reduce_sum(tf.square(pos_head_e + pos_rel_e - pos_tail_e), axis=1)
    neg = tf.reduce_sum(tf.square(neg_head_e + neg_rel_e - neg_tail_e), axis=1)    
            
    loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))  

    return loss
# def pred_loss(
#     pos_head_e,
#     pos_rel_e,
#     pos_tail_e,
#     neg_head_e,
#     neg_rel_e,
#     neg_tail_e,
#     margin=2
#     ):
#     pos = tf.reduce_sum(tf.multiply(tf.multiply(pos_head_e,pos_rel_e),pos_tail_e))
#     neg = tf.reduce_sum(tf.multiply(tf.multiply(neg_head_e,neg_rel_e),neg_tail_e))

#     return tf.reduce_sum(tf.maximum(pos-neg + margin,0))

def link_score(head_e, rel_e, tail_e):
    return -np.sum(np.square(head_e + rel_e - tail_e))

def exp_score(triple,k,data,entity_embeddings,relation_embeddings):

    '''Compute score for explanations. Returns k explanations 
        for each triple in <dataset>
    '''    
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

def predict_link(
    head_e,
    tail_e,
    rel2idx,
    relations_str,
    relation_embeddings,
    score_fun=link_score,
    k=1):
    '''Predicts link between two nodes'''

    scores = []

    for rel in relations:

        rel_idx = rel2idx[rel]
        rel_e = relation_embeddings[rel_idx]

        score = score_fun(head_e,rel_e,tail_e)

        scores.append((rel,rel_idx,score))

    return sorted(scores,key=lambda x:x[2],reverse=True)[:k]
