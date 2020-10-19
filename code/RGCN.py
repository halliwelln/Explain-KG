#!/usr/bin/env python3

import tensorflow as tf
import utils
import numpy as np
import random as rn
import os
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow.python.ops import embedding_ops

class RGCN_Layer(tf.keras.layers.Layer):
    def __init__(self,num_relations,output_dim,**kwargs):
        super(RGCN_Layer,self).__init__(**kwargs)
        self.num_relations = num_relations
        self.output_dim = output_dim
        
    def build(self,input_shape):
        
        input_dim = int(input_shape[0][-1])
        
        self.relation_kernels = [
                self.add_weight(
                    shape=(input_dim, self.output_dim),
                    name="relation_kernels",
                    trainable=True,
                    initializer="random_normal",
                )
                for _ in range(self.num_relations)
            ]

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            name="self_kernel",
            trainable=True,
            initializer="random_normal",
        )
        
    def call(self, inputs):
        
        features, *A_mats = inputs

        output = tf.matmul(features,self.self_kernel)
        
        for i in range(self.num_relations):
            
            h = tf.tensordot(A_mats[i], features,axes=1)
            output += tf.tensordot(h,self.relation_kernels[i],axes=1)
            
        return output

class DistMult(tf.keras.layers.Layer):
    def __init__(self, num_relations,**kwargs):
        super(DistMult,self).__init__(**kwargs)
        self.num_relations = num_relations
        
    def build(self,input_shape):
        
        embedding_dim = input_shape[0][-1]
        
        self.kernel = self.add_weight(
            shape=(self.num_relations,embedding_dim),
            trainable=True,
            initializer='random_normal',
            name='rel_embedding'
        )
        
    def call(self,inputs):
        
        head_e,rel_idx,tail_e = inputs
        
        rel_e = embedding_ops.embedding_lookup_v2(self.kernel,rel_idx)
        
        return tf.reduce_sum(head_e*rel_e*tail_e, axis=-1)

if __name__ == '__main__':

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    data = np.load(os.path.join('..','data','royalty.npz'))

    entities = data['entities'].tolist()
    num_entities = len(entities)
    relations = data['relations'].tolist()
    num_relations = len(relations)
    embedding_dim = 3
    ent2idx = dict(zip(entities, range(num_entities)))
    rel2idx = dict(zip(relations, range(num_relations)))