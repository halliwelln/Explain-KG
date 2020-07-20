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

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)

triples = [('Eve', 'type', 'Lecturer'),
           ('Eve', 'type', 'Person'), 
           ('Lecturer', 'subClassOf', 'Person'), 
           ('David', 'type', 'Person'),
           ('David', 'type', 'Researcher'),
           ('Researcher', 'subClassOf', 'Person'),
           ('Flora', 'hasSpouse', 'Gaston'),
           ('Gaston', 'type', 'Person'),
           ('Flora', 'type', 'Person')
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

train2idx = utils.train2idx(train,ent2idx,rel2idx)

embedding_dim = 10
s1 = 1
s2 = 1.5
learning_rate = .001
max_iter = 100
gamma = (1/(s1**2)) - (1/(s2**2))

A = utils.get_adjacency_matrix(train,entities,num_entities)
indices = utils.get_neighbor_idx(A)

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
    
    normal_s1 = halfnorm.rvs(loc=0,scale=s1,size=1)
    normal_s2 = halfnorm.rvs(loc=0,scale=s2,size=1)
    
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

            hessian += p_diff_mat - h
            
    return hessian

def explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A):

    hessian = get_hessian(i,s1,s2,gamma,X,A,embedding_dim)
    pij = get_pij(i=i,j=j,s1=s1,s2=s2,prior=prior, X=X)

    invert = (-hessian) / ((gamma**2 * (pij) * (1-pij)) + .00001)

    hess_inv = np.linalg.inv(hessian)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]
    x_l = X[l,:]

    xij_diff = (x_i - x_j).reshape(1,-1)

    xlk_diff = (x_l - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xlk_diff).squeeze()

explanations = []

for i,_,j in train2idx:

    for k,_,l in train2idx:

        if (i != k) and (j != l):

            print(i,j,k,l,explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A))

#add jaccard score
