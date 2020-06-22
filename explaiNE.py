#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pandas as pd
import random as rn
import os
import cne
import maxent
from scipy import sparse
from scipy.stats import halfnorm

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

triples = [('Eve', 'type', 'Lecturer'),
           #('Eve', 'type', 'Person'), 
           ('Lecturer', 'subClassOf', 'Person'), 
           #('David', 'type', 'Person'),
           ('David', 'type', 'Researcher'),
           ('Researcher', 'subClassOf', 'Person'),
           ('Flora', 'hasSpouse', 'Gaston'),
           ('Gaston', 'type', 'Person'),
           #('Flora', 'type', 'Person')
          ]

train = np.array(triples)

entities = np.unique(np.concatenate((train[:,0], train[:,2]), axis=0)).tolist()
relations = np.unique(train[:,1]).tolist()

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

idx2ent = {idx:ent for ent,idx in ent2idx.items()}
idx2rel = {idx:rel for rel,idx in rel2idx.items()}

embedding_dim = 10

A = np.zeros(shape=(num_entities,num_entities))

for h,r,t in train:
    
    h_idx = entities.index(h)
    r_idx = relations.index(r)
    t_idx = entities.index(t)
    
    A[h_idx, t_idx] = 1

A_sparse = sparse.csr_matrix(A)  
prior = maxent.BGDistr(A_sparse) 
prior.fit()

CNE = cne.ConditionalNetworkEmbedding(
    A=A_sparse,
    d=embedding_dim,
    s1=1,
    s2=1.5,
    prior_dist=prior
    )
CNE.fit(lr=.001, max_iter=100)

X = CNE._ConditionalNetworkEmbedding__emb

def get_pij(i,j,s1,s2,prior, X):
    
    p_prior = prior.get_row_probability([i], [j])
    
    normal_s1 = halfnorm.rvs(loc=0,scale=s1,size=1)
    normal_s2 = halfnorm.rvs(loc=0,scale=s2,size=1)
    
    numerator = p_prior * normal_s1
    denom = numerator + (1-p_prior)*normal_s2
    
    return numerator/denom

i=0
s1 = 1
s2 = 1.5
gamma = (1/(s1**2)) - (1/(s2**2))

def get_hessian(i,s1,s2,gamma,X,A,embedding_dim):
    
    hessian = np.zeros(shape=(embedding_dim,embedding_dim))

    for j in range(A.shape[0]):

        if i != j:

            x_i = X[i,:]
            x_j = X[j,:]

            x_diff = (x_i - x_j).reshape(-1,1)  
            
            prob = get_pij(i,j,s1,s2,prior, X)

            h = (gamma**2) * np.dot(x_diff,x_diff.T) * (prob * (1-prob))

            a = A[i,j]

            p_diff = gamma * (prob - a)[0]

            p_diff_mat = p_diff * np.identity(h.shape[0])

            hessian += p_diff_mat - h
            
    return hessian

j=1
k=2

def get_gradient(i,j,k,s1,s2,embedding_dim,gamma,X,A):

    hessian = get_hessian(i,s1,s2,gamma,X,A,embedding_dim)
    pij = get_pij(i=i,j=j,s1=s1,s2=s2,prior=prior, X=X)

    invert = (-hessian) / (gamma**2 * (pij) * (1-pij))

    hess_inv = np.linalg.inv(-hessian)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]

    xij_diff = (x_i - x_j).reshape(1,-1)

    xik_diff = (x_i - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xik_diff).squeeze()

print(get_gradient(i,j,k,s1,s2,embedding_dim,gamma,X,A))