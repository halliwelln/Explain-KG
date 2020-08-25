#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random as rn
import os
import cne
import maxent
from scipy.stats import halfnorm
import utils
import joblib
# import cupyx.scipy.sparse
# import cupy as np
#export PATH=$PATH:/Users/nhalliwe/.local/bin
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

print(f'CPU count: {joblib.cpu_count()}')

data = np.load(os.path.join('.','data','royalty.npz'))

train = data['X_train']
test = data['X_test']

train_exp = data['train_exp']
test_exp = data['test_exp']

full_train = np.concatenate((train,train_exp), axis=0)

entities = data['entities'].tolist()
relations = data['relations'].tolist()

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

idx2ent = {idx:ent for ent,idx in ent2idx.items()}
idx2rel = {idx:rel for rel,idx in rel2idx.items()}

train2idx = utils.array2idx(full_train,ent2idx,rel2idx)
test2idx = utils.array2idx(test,ent2idx,rel2idx)

#adjacency_data = np.concatenate((full_train,test,test_exp), axis=0)
#adjacency_data = np.concatenate((full_train,test), axis=0)
#A = cupyx.scipy.sparse.csr_matrix(utils.get_adjacency_matrix(full_train,entities,num_entities))
A = utils.get_adjacency_matrix(full_train,entities,num_entities)

train_exp = [[(ent2idx[h],ent2idx[t])] for h,_,t in train_exp]
test_exp = [[(ent2idx[h],ent2idx[t])] for h,_,t in test_exp]

embedding_dim = 50
s1 = 1
s2 = 1.5
learning_rate = .001
max_iter = 2
gamma = (1/(s1**2)) - (1/(s2**2))
top_k = 1

prior = maxent.BGDistr(A) 
prior.fit()

CNE = cne.ConditionalNetworkEmbedding(
    A=A,
    d=embedding_dim,
    s1=s1,
    s2=s2,
    prior_dist=prior
    )

CNE.fit(lr=learning_rate, max_iter=max_iter)

X = CNE._ConditionalNetworkEmbedding__emb

def get_pij(i,j,s1,s2,prior, X):
    
    p_prior = prior.get_row_probability([i], [j])
    
    normal_s1 = halfnorm.rvs(loc=0,scale=s1,size=1,random_state=SEED)
    normal_s2 = halfnorm.rvs(loc=0,scale=s2,size=1,random_state=SEED)
    
    numerator = p_prior * normal_s1
    denom = numerator + (1-p_prior)*normal_s2
    
    return numerator/denom

@profile
def get_hessian(i,s1,s2,gamma,X,A,embedding_dim):
    
    hessian = np.zeros(shape=(embedding_dim,embedding_dim))

    for j in range(A.shape[0]):

        if i != j:

            x_i = X[i,:]
            x_j = X[j,:]

            x_diff = (x_i - x_j).reshape(-1,1)  
            
            prob = get_pij(i,j,s1,s2,prior, X)

            h = (gamma**2) * np.dot(x_diff,x_diff.T) * (prob * (1-prob))

            p_diff_mat = gamma * (prob - A[i,j])[0] * np.identity(h.shape[0])

            hessian += (p_diff_mat - h)
            
    return hessian
@profile
def explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A):

    hessian = get_hessian(i=i,s1=s1,s2=s2,gamma=gamma,X=X,A=A,embedding_dim=embedding_dim)
    pij = get_pij(i=i,j=j,s1=s1,s2=s2,prior=prior, X=X)

    invert = (-hessian) / ((gamma**2 * (pij) * (1-pij)))

    hess_inv = np.linalg.inv(invert)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]
    x_l = X[l,:]

    xij_diff = (x_i - x_j).T

    xlk_diff = (x_l - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xlk_diff).squeeze()

# def loop(i,j,k,l,s1,s2,embedding_dim,gamma,X,A, train2idx):

#     score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A)

#     return ((k,l),score)
@profile
def get_explanations(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,train2idx):

    row,col = A.nonzero()

    neighbors = []

    for idx, l in enumerate(col):
        if l == i:

            if (i,j) != (row[idx],col[idx]):

                neighbors.append([row[idx],col[idx]])

    if len(neighbors) > top_k:

        temp = []

        for k,l in neighbors:

            score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A)

            temp.append(((k,l),score))

        # temp = joblib.Parallel(n_jobs=-2,verbose=20)(joblib.delayed(loop)(i,j,
        #     k,l,s1,s2,embedding_dim,gamma,X,A, train2idx) for k,l in neighbors)

    else:

        temp = []

        for k,_,l in train2idx:

            if (i,j) != (k,l):

                score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A)

                temp.append(((k,l),score))

        # temp = joblib.Parallel(n_jobs=-2,verbose=20)(joblib.delayed(loop)(i,j,
        #     k,l,s1,s2,embedding_dim,gamma,X,A, train2idx) for k,_,l in train2idx if (i,j) != (k,l))

    sorted_scores = sorted(temp,key=lambda x:x[1], reverse=True)[0:top_k]

    explanation = [tup for tup,_ in sorted_scores]

    return explanation

# explanations = joblib.Parallel(n_jobs=-2, verbose=20)(
#     joblib.delayed(get_explanations)(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,train2idx) for i,_,j in test2idx[0:2]
#     )

explanations = [get_explanations(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,train2idx) for i,_,j in test2idx[0:2]]

# print(explanations[0:5])
# print(test_exp[0:5])

jaccard = utils.jaccard_score(explanations,test_exp[0:2])

print(f"Jaccard score={jaccard} using:")
print(f"embedding dimensions={embedding_dim},s1={s1},s2={s2}")
print(f"learning_rate={learning_rate},max_iter={max_iter}")
