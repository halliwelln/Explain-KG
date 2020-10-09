#!/usr/bin/env python3

import tensorflow as tf
import utils
import numpy as np
import random as rn
import os

class GCN(tf.keras.layers.Layer):
    def __init__(self,units,**kwargs):
        super(GCN,self).__init__(**kwargs)
        self.units=units
        
    def build(self,input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[-1][-1],self.units),
            trainable=True,
            name='kernel',
            initializer=tf.keras.initializers.RandomNormal(seed=SEED)
        )

    def call(self,inputs):

        D_hat_inv,A_hat,H = inputs

        DHW = tf.keras.backend.dot(D_hat_inv,tf.keras.backend.dot(H,self.kernel))

        output = tf.keras.backend.dot(D_hat_inv,tf.keras.backend.dot(A_hat,DHW))

        return output
    
    def get_config(self):
        base_config = super(GCN, self).get_config()
        config = {'units': self.units}
        return dict(list(base_config.items()) + list(config.items()))

def gcn_model(num_entities,num_features,output_dim):

    feature_input = tf.keras.layers.Input(shape=(num_features,),sparse=False,name='feature_input')
    adjacency_input = tf.keras.layers.Input(shape=(num_entities,),sparse=True,name='adjacency_input')
    degree_input = tf.keras.layers.Input(shape=(num_entities,),sparse=True,name='degree_input')

    gcn_layer = GCN(output_dim,name='gcn_layer')([degree_input,adjacency_input,feature_input])

    activation = tf.keras.layers.Activation('sigmoid',name='activation')(gcn_layer)
    
    model = tf.keras.Model(
        inputs=[degree_input,adjacency_input,feature_input],
        outputs=activation
    )

    return model

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    data = np.load(os.path.join('.','data','royalty.npz'))

    triples = data['triples']
    traces = data['traces']
    entities = data['entities'].tolist()
    num_entities = len(entities)
    relations = data['relations'].tolist()
    num_relations = len(relations)
    num_features = 5

    indices = []
    self_loop = [[i,i] for i in range(num_entities)]

    for h,_,t in triples:
        h_idx = entities.index(h)
        t_idx = entities.index(t)
        indices.append([h_idx,t_idx])

    indices += self_loop

    indices = np.unique(indices,axis=0).tolist()

    values = np.ones(len(indices))

    A_hat_sparse = tf.sparse.reorder(
        tf.sparse.SparseTensor(indices,values,
            dense_shape=[num_entities,num_entities]
        )
    )

    diag = tf.sparse.reduce_sum(A_hat_sparse,axis=0)

    D_hat = tf.sparse.reorder(
        tf.sparse.SparseTensor(self_loop,diag,dense_shape=[num_entities,num_entities]
        )
    ) 

    X = tf.convert_to_tensor(np.random.rand(num_entities,num_features))

    y = np.array([relations.index(i) for i in triples[:,1]])

    train_idx, test_idx = train_test_split(range(num_entities),
        random_state=SEED,test_size=0.33)

    train_mask = np.zeros(num_entities,dtype=bool)
    train_mask[train_idx] = 1

    test_mask = np.zeros(num_entities,dtype=bool)
    test_mask[test_idx] = 1

    y_train = tf.keras.utils.to_categorical(y[train_mask],num_classes=num_relations)

    y_test = tf.keras.utils.to_categorical(y[test_mask],num_classes=num_relations)
    print(X.shape)
    print(y.shape)
    print(y_train.shape)
    print(y_test.shape)
    # model = gcn_model(num_entities,num_features,output_dim=num_entities)

    # model.compile(optimizer='sgd',loss='categorical_crossentropy')

    # model.fit(
    #     x=[D_hat,A_hat_sparse,X],
    #     y=y_train,
    #     sample_weight=train_mask,
    #     batch_size=num_entities,
    #     epochs=1,
    #     shuffle=False
    # )

    # preds = model.predict([D_hat,A_hat_sparse,X],batch_size=num_entities)

    # preds = np.argmax(preds,axis=1)[test_mask]

    # y_test = np.argmax(y_test,axis=1)

    # acc = accuracy_score(y_test,preds)

    # print(f"Test set accurate: {acc}")

