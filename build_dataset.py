#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import utils
from sklearn.model_selection import train_test_split

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

MAX_TRACES = 2

spouse_triples,spouse_traces = utils.parse_ttl(
    file_name=os.path.join('.','data','traces','spouse.ttl'),
    max_traces=MAX_TRACES)

print(f"number of triples {len(spouse_triples)}")

#####################################
X_train, X_test, train_exp, test_exp = train_test_split(spouse_triples,
    spouse_traces,test_size=0.30, random_state=42)

# #remove triples with unseen entities from test set and add to train set
# train_entities = np.unique(np.concatenate((X_train[:,0], X_train[:,2],
#     train_exp[:,0],train_exp[:,2]), axis=0))
# #test_entities = np.unique(np.concatenate((X_test[:,0],X_test[:,2],test_exp[:,0], test_exp[:,2]), axis=0))

# unseen = []
# for h,r,t in X_test:
#     if h not in train_entities or t not in train_entities:
#         unseen.append(True)
#     else:
#         unseen.append(False)
        
# unseen = np.array(unseen)

# X_train = np.concatenate((X_train,X_test[unseen]),axis=0)
# train_exp = np.concatenate((train_exp,test_exp[unseen]), axis=0)

# X_test = X_test[~unseen]
# test_exp = test_exp[~unseen]

# print(f"Training set size {X_train.shape}")
# print(f"Test set size {X_test.shape}")
# #####################################

exp_entities = np.array([[spouse_traces[:,i][:,0],spouse_traces[:,i][:,2]] for i in range(max_traces)]).flatten()
exp_entities = np.array([i for i in exp_entities if i != "0.0"])

exp_relations = np.array([spouse_traces[:,i][:,1] for i in range(max_traces)]).flatten()
exp_relations = np.array([i for i in exp_relations if i != "0.0"])

entities = np.unique(np.concatenate([spouse_triples[:,0], spouse_triples[:,2], exp_entities],axis=0))
relations = np.unique(np.concatenate([spouse_triples[:,1], exp_relations],axis=0))

np.savez(os.path.join('.','data','royalty_spouse.npz'),
    X_train=X_train, X_test=X_test, train_exp=train_exp,
    test_exp=test_exp,entities=entities,relations=relations)

# np.savez(os.path.join('.','data','royalty.npz'),
#     triples=royalty,
#     explanations=explanations,
#     entities=entities,
#     relations=relations
#     )

