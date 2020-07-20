#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model

def build_model(
    embedding_size,
    num_entities,
    num_relations,
    batch_size,
    num_epochs,
    margin,
    sqrt_size,
    seed
    ):

    pos_head_input = Input(shape=(1,), name='pos_head_input')
    neg_head_input = Input(shape=(1,), name='neg_head_input')
    pos_tail_input = Input(shape=(1,), name='pos_tail_input')
    neg_tail_input = Input(shape=(1,), name='neg_tail_input')
    relation_input = Input(shape=(1,), name='relation_input')

    entity_embedding = Embedding(
        input_dim=num_entities,
        output_dim=embedding_size,
        name='entity_embeddings',
        embeddings_initializer=RandomUniform(minval=-sqrt_size,maxval=sqrt_size,seed=seed)
        )

    relation_embedding = Embedding(
        input_dim=num_relations,
        output_dim=embedding_size,
        name='relation_embeddings',
        embeddings_initializer=RandomUniform(minval=-sqrt_size,maxval=sqrt_size,seed=seed)
        )

    pos_head_e = entity_embedding(pos_head_input)
    neg_head_e = entity_embedding(neg_head_input)
    pos_tail_e = entity_embedding(pos_tail_input)
    neg_tail_e = entity_embedding(neg_tail_input)
    rel_e = relation_embedding(relation_input)

    model = Model(
        inputs=[
            pos_head_input,
            neg_head_input, 
            pos_tail_input, 
            neg_tail_input, 
            relation_input
            ], 
        outputs=[
            pos_head_e,
            neg_head_e, 
            pos_tail_e, 
            neg_tail_e, 
            rel_e
            ]
        )

    return model