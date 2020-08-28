#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import utils
import rdflib
from sklearn.model_selection import train_test_split

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

g = rdflib.Graph()
g.parse(os.path.join('.','data','sparql'),format="xml")

spouse_traces = utils.parse_traces(os.path.join('.','data','traces','spouse.ttl'))

full_royalty = []

for subj,pred,obj in g:
    
    s = subj.split('/')[-1]
    p = pred.split('/')[-1]
    o = obj.split('/')[-1]
    
    if 'with_no_name_entry' in s or 'with_no_name_entry' in o:
        continue
        
    full_royalty.append((s,p,o))

royalty = []
explanations = []

for k in full_royalty:
    if spouse_traces[k]:
        royalty.append(k)
        explanations.append(spouse_traces[k])

royalty = np.array(royalty)
explanations = np.array(explanations).reshape(-1,3)

print(f"number of total triples {len(full_royalty)}")
print(f"number of triples with explanations {len(royalty)}")

#####################################
X_train, X_test, train_exp, test_exp = train_test_split(royalty, explanations, test_size=0.30, random_state=42)

#remove triples with unseen entities from test set and add to train set
train_entities = np.unique(np.concatenate((X_train[:,0], X_train[:,2], train_exp[:,0],train_exp[:,2]), axis=0))
#test_entities = np.unique(np.concatenate((X_test[:,0],X_test[:,2],test_exp[:,0], test_exp[:,2]), axis=0))

unseen = []
for h,r,t in X_test:
    if h not in train_entities or t not in train_entities:
        unseen.append(True)
    else:
        unseen.append(False)
        
unseen = np.array(unseen)

X_train = np.concatenate((X_train,X_test[unseen]),axis=0)
train_exp = np.concatenate((train_exp,test_exp[unseen]), axis=0)

X_test = X_test[~unseen]
test_exp = test_exp[~unseen]

print(f"Training set size {X_train.shape}")
print(f"Test set size {X_test.shape}")
#####################################

entities = np.unique(np.concatenate((royalty[:,0],royalty[:,2],
    explanations[:,0], explanations[:,2]), axis=0))

relations = np.unique(np.concatenate((royalty[:,1],explanations[:,1]), axis=0))

np.savez(os.path.join('.','data','royalty.npz'),
    X_train=X_train, X_test=X_test, train_exp=train_exp,
    test_exp=test_exp,entities=entities,relations=relations)

# np.savez(os.path.join('.','data','royalty.npz'),
#     triples=royalty,
#     explanations=explanations,
#     entities=entities,
#     relations=relations
#     )

