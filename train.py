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
NUM_EPOCHS = 10
MARGIN = 2
LEARNING_RATE = .001

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    model = transE.ExTransE(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_size=EMBEDDING_SIZE,
        margin=MARGIN,
        random_state=SEED)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE))

model.fit(
    x=[train2idx[:,0],
    train2idx[:,1],
    train2idx[:,2],
    trainexp2idx[:,:,0].reshape(-1),
    trainexp2idx[:,:,1].reshape(-1),
    trainexp2idx[:,:,2].reshape(-1)],
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE
    )

# model = transE.ExTransE(
#     num_entities=NUM_ENTITIES,
#     num_relations=NUM_RELATIONS,
#     embedding_size=EMBEDDING_SIZE,
#     margin=MARGIN,
#     random_state=SEED)
# optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
# train_data = tf.data.Dataset.from_tensor_slices((train2idx[:,0],train2idx[:,1],train2idx[:,2],
#                                                 trainexp2idx[:,:,0].reshape(-1),trainexp2idx[:,:,1].reshape(-1),
#                                                  trainexp2idx[:,:,2].reshape(-1))).batch(BATCH_SIZE)

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

#             explain_loss = transE.exp_loss(
#                 pos_head_exp_e,
#                 pos_rel_exp_e,
#                 pos_tail_exp_e,
#                 neg_head_exp_e,
#                 neg_rel_exp_e,
#                 neg_tail_exp_e,
#                 margin=MARGIN
#             )

#             total_loss = prediction_loss + explain_loss

#         grads = tape.gradient(total_loss,model.trainable_variables)
#         optimizer.apply_gradients(zip(grads,model.trainable_variables))

#     epoch_loss.append(np.round(total_loss.numpy(),5))

# print(np.mean(epoch_loss))