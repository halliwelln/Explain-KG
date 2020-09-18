#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import utils
from sklearn.model_selection import train_test_split
import argparse

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('trace_file',type=str)
parser.add_argument('max_padding', type=int)
args = parser.parse_args()

TRACE_FILE = args.trace_file
MAX_PADDING = args.max_padding

# spouse_triples,spouse_traces = utils.parse_ttl(
#     file_name=os.path.join('.','data','traces','spouse.ttl'),
#     max_padding=MAX_PADDING)

# print(f"number of triples {len(spouse_triples)}")

# X_train, X_test, train_exp, test_exp = train_test_split(spouse_triples,
#     spouse_traces,test_size=0.30, random_state=42)

# exp_entities = np.array([[spouse_traces[:,i,:][:,0],
#     spouse_traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

# exp_relations = np.array([spouse_traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

# entities = np.unique(np.concatenate([spouse_triples[:,0], spouse_triples[:,2], exp_entities],axis=0))
# relations = np.unique(np.concatenate([spouse_triples[:,1], exp_relations],axis=0))

# np.savez(os.path.join('.','data','royalty_spouse.npz'),
#     X_train=X_train, X_test=X_test, train_exp=train_exp,
#     test_exp=test_exp,entities=entities,relations=relations)

triples,traces = utils.parse_ttl(
    file_name=os.path.join('.','data','traces',TRACE_FILE+'.ttl'),
    max_padding=MAX_PADDING)

print(f"number of triples {len(triples)}")

X_train, X_test, train_exp, test_exp = train_test_split(triples,
    traces,test_size=0.30, random_state=42)

exp_entities = np.array([[traces[:,i,:][:,0],
    traces[:,i,:][:,2]] for i in range(MAX_PADDING)]).flatten()

exp_relations = np.array([traces[:,i,:][:,1] for i in range(MAX_PADDING)]).flatten()

entities = np.unique(np.concatenate([triples[:,0], triples[:,2], exp_entities],axis=0))
relations = np.unique(np.concatenate([triples[:,1], exp_relations],axis=0))

np.savez(os.path.join('.','data','royalty_'+TRACE_FILE+'.npz'),
    X_train=X_train, X_test=X_test, train_exp=train_exp,
    test_exp=test_exp,entities=entities,relations=relations)

print('Dataset built.')








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
#merging dataset for all rules, remember to pad!



