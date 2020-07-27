#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random as rn
import os
import cne
import maxent
from scipy import sparse
from scipy.stats import halfnorm
import utils
import joblib

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

data = np.load('/Users/nhalliwe/Desktop/Explain-KG/data/royalty.npz')

train = np.concatenate((data['train'][0:10],data['exp_train'][0:10]),axis=0)

entities = np.unique(np.concatenate((train[:,0], train[:,2]), axis=0)).tolist()

relations = np.unique(train[:,1]).tolist()

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

idx2ent = {idx:ent for ent,idx in ent2idx.items()}
idx2rel = {idx:rel for rel,idx in rel2idx.items()}

train2idx = utils.train2idx(train,ent2idx,rel2idx)

train_exp = []

for h,_,t in data['exp_train'][0:10]:

    train_exp.append([(ent2idx[h],ent2idx[t])])

embedding_dim = 20
s1 = 1
s2 = 1.5
learning_rate = .001
max_iter = 100
gamma = (1/(s1**2)) - (1/(s2**2))
top_k = 1

A = utils.get_adjacency_matrix(train,entities,num_entities)

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

def get_hessian(i,s1,s2,gamma,X,A,embedding_dim):
    
    hessian = np.zeros(shape=(embedding_dim,embedding_dim))

    for j in range(A.shape[0]):

        if i != j:

            x_i = X[i,:]
            x_j = X[j,:]

            x_diff = (x_i - x_j).reshape(-1,1)  
            
            prob = get_pij(i,j,s1,s2,prior, X)

            h = (gamma**2) * np.dot(x_diff,x_diff.T) * (prob * (1-prob))

            p_diff = gamma * (prob - A[i,j])[0]

            p_diff_mat = p_diff * np.identity(h.shape[0])

            hessian += (p_diff_mat - h)
            
    return hessian

def explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A):

    hessian = get_hessian(i=i,s1=s1,s2=s2,gamma=gamma,X=X,A=A,embedding_dim=embedding_dim)
    pij = get_pij(i=i,j=j,s1=s1,s2=s2,prior=prior, X=X)

    invert = (-hessian) / ((gamma**2 * (pij) * (1-pij)) + .0001)

    hess_inv = np.linalg.inv(invert)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]
    x_l = X[l,:]

    xij_diff = (x_i - x_j).T

    xlk_diff = (x_l - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xlk_diff).squeeze()

def get_explanations(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,data):

    temp = []

    for k,_,l in data:

        if (i,j) != (k,l):

            score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A)

            temp.append(((k,l),score))

    sorted_scores = sorted(temp,key=lambda x:x[1], reverse=True)[0:top_k]

    explanation = [tup for tup,_ in sorted_scores]

    return explanation

explanations = joblib.Parallel(n_jobs=-2, verbose=20)(
    joblib.delayed(get_explanations)(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,train2idx) for i,_,j in train2idx
    )

# explanations = []

# for i,_,j in train2idx:

#     temp = []

#     for k,_,l in train2idx:
        
#         if (i,j) != (k,l):

#             score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A)

#             temp.append(((k,l),score))

#     sorted_scores = sorted(temp,key=lambda x:x[1], reverse=True)[0:top_k]

#     explanation_list = [tup for tup,_ in sorted_scores]

#     explanations.append(explanation_list)

for i in range(len(explanations[0:10])):
    print('i',i)
    print('train',train2idx[i])
    print('pred exp',explanations[i])
    print('true exp',train_exp[i])

print(utils.jaccard_score(explanations[0:10],train_exp))
