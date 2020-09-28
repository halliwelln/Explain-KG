#!/usr/bin/env python3

import tensorflow as tf
import transE
import utils
import numpy as np
import random as rn
import os

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

data = np.load(os.path.join('.','data','royalty_spouse.npz'))

train = data['X_train']
test = data['X_test']

train_exp = data['train_exp']
test_exp = data['test_exp']

entities = data['entities'].tolist()
relations = data['relations'].tolist()

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

#idx2ent = {idx:ent for ent,idx in ent2idx.items()}
#idx2rel = {idx:rel for rel,idx in rel2idx.items()}

train2idx = utils.array2idx(train,ent2idx,rel2idx)
test2idx = utils.array2idx(test,ent2idx,rel2idx)

trainexp2idx = utils.array2idx(train_exp,ent2idx,rel2idx)
testexp2idx = utils.array2idx(test_exp,ent2idx,rel2idx)

EMBEDDING_SIZE = 50
BATCH_SIZE = 128
NUM_EPOCHS = 300
MARGIN = 2
LEARNING_RATE = .001

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    model = transE.ExTransE(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_size=EMBEDDING_SIZE,
        random_state=SEED
        )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        margin=MARGIN,
        pred_loss=transE.pred_loss,
        exp_loss=transE.exp_loss
        )

model.fit(
    x=[
        train2idx[:,0],
        train2idx[:,1],
        train2idx[:,2],
        trainexp2idx[:,:,0].flatten(),
        trainexp2idx[:,:,1].flatten(),
        trainexp2idx[:,:,2].flatten()
    ],
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE
    )

# train_data = tf.data.Dataset.from_tensor_slices(
#     (train2idx[:,0],train2idx[:,1],train2idx[:,2],
#     trainexp2idx[:,:,0],trainexp2idx[:,:,1],trainexp2idx[:,:,2])).batch(BATCH_SIZE)

# optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
# model = transE.ExTransE(NUM_ENTITIES,NUM_RELATIONS,EMBEDDING_SIZE,random_state=SEED)

# epoch_loss = []

# for epoch in range(NUM_EPOCHS):

#     for pos_head, pos_rel, pos_tail, pos_head_exp,pos_rel_exp, pos_tail_exp in train_data:

#         neg_head, neg_tail = utils.get_negative_triples(
#             head=pos_head, 
#             rel=pos_rel, 
#             tail=pos_tail,
#             num_entities=NUM_ENTITIES
#             )

#         neg_head_exp, neg_tail_exp = utils.get_negative_triples(
#             head=pos_head_exp, 
#             rel=pos_rel_exp, 
#             tail=pos_tail_exp,
#             num_entities=NUM_ENTITIES
#             )

#         with tf.GradientTape() as tape:

#             pos_head_e,pos_rel_e,pos_tail_e,pos_head_exp_e,pos_rel_exp_e,pos_tail_exp_e = model([
#                 pos_head,
#                 pos_rel,
#                 pos_tail,
#                 pos_head_exp,
#                 pos_rel_exp,
#                 pos_tail_exp
#                 ]
#             )

#             neg_head_e,neg_rel_e,neg_tail_e,neg_head_exp_e,neg_rel_exp_e,neg_tail_exp_e = model([
#                 neg_head,
#                 pos_rel,#pos_rel is correct, 
#                 neg_tail,
#                 neg_head_exp,
#                 pos_rel_exp,
#                 neg_tail_exp
#                 ]
#             )

#             prediction_loss = transE.pred_loss(
#                 pos_head_e,
#                 pos_rel_e,
#                 pos_tail_e,
#                 neg_head_e,
#                 neg_rel_e,
#                 neg_tail_e,
#                 margin=MARGIN
#             )

#             # explain_loss = self.exp_loss(
#             #     pos_head_exp_e,
#             #     pos_rel_exp_e,
#             #     pos_tail_exp_e,
#             #     neg_head_exp_e,
#             #     neg_rel_exp_e,
#             #     neg_tail_exp_e,
#             #     margin=self.margin
#             # )
#             explain_loss = transE.exp_loss(
#                 pos_head_e,
#                 pos_rel_e, 
#                 pos_tail_e,
#                 pos_head_exp_e,
#                 pos_rel_exp_e,
#                 pos_tail_exp_e
#                 )

#             total_loss = prediction_loss + explain_loss

#         grads = tape.gradient(total_loss,model.trainable_variables)
#         optimizer.apply_gradients(zip(grads,model.trainable_variables))
        
#     epoch_loss.append(np.round(total_loss.numpy(),5))

# print("mean loss",np.mean(epoch_loss))

test_head_e, test_rel_e, test_tail_e, test_exp_head_e, test_exp_rel_e, test_exp_tail_e = model.predict(x=[
        test2idx[:,0],
        test2idx[:,1],
        test2idx[:,2],
        testexp2idx[:,:,0].flatten(),
        testexp2idx[:,:,1].flatten(),
        testexp2idx[:,:,2].flatten()
        ]
    )

top_k = 1
pred_exp = []

for i in range(len(test2idx)):
    
    triple_h_e = test_head_e[i]
    triple_r_e = test_rel_e[i]
    triple_t_e = test_tail_e[i]
    
    squared_diff = np.square(triple_h_e - test_exp_head_e) + np.square(triple_r_e-test_exp_rel_e) + np.square(triple_t_e-test_exp_tail_e)

    l2_dist = np.sqrt(np.sum(squared_diff,axis=1))

    closest_l2 = np.argsort(l2_dist)[:top_k]

    k_closest = testexp2idx[closest_l2]

    pred_exp.append(k_closest)

pred_exp = np.array(pred_exp).reshape(-1,1,3)

print(utils.jaccard_score(testexp2idx,pred_exp))

# entity_embeddings = utils.get_entity_embeddings(model)
# relation_embeddings = utils.get_relation_embeddings(model)

# top_k = 1
# pred_exp = []

# for i in range(len(test2idx)):

#     h_idx, r_idx, t_idx = test2idx[i]

#     triple_h_e = entity_embeddings[h_idx]
#     triple_r_e = relation_embeddings[r_idx]
#     triple_t_e = entity_embeddings[t_idx]

#     h_e = entity_embeddings[testexp2idx[:,:,0].flatten()]
#     r_e = relation_embeddings[testexp2idx[:,:,1].flatten()]
#     t_e = entity_embeddings[testexp2idx[:,:,2].flatten()]
    
#     squared_diff = np.square(triple_h_e - h_e) + np.square(triple_r_e-r_e) + np.square(triple_t_e-t_e)

#     l2_dist = np.sqrt(np.sum(squared_diff,axis=1))

#     closest_l2 = np.argsort(l2_dist)[:top_k]

#     k_closest = testexp2idx[closest_l2]

#     pred_exp.append(k_closest) 

# pred_exp = np.array(pred_exp).reshape(-1,top_k,3)