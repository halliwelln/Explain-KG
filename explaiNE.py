#!/usr/bin/env python3

import numpy as np
import random as rn
import os
import cne
import maxent
from scipy.stats import halfnorm
import utils
import joblib

def get_pij(i,j,X,s1,s2,prior,seed):
    
    p_prior = prior.get_row_probability([i],[j])[0]

    x_i = X[i,:]
    x_j = X[j,:]

    diff = np.linalg.norm(x_i-x_j)

    normal_s1 = halfnorm.rvs(loc=diff,scale=s1,size=1,random_state=seed)[0]
    normal_s2 = halfnorm.rvs(loc=diff,scale=s2,size=1,random_state=seed)[0]
    
    numerator = p_prior * normal_s1
    denom = numerator + (1-p_prior)*normal_s2
    
    return numerator/denom

def compute_prob(i,s1,s2,X,A,prior,seed):
    prob = []
    for j in range(A.shape[0]):
        prob.append(get_pij(i,j,X,s1,s2,prior,seed))
    return prob

def get_hessian(i,s1,s2,gamma,X,A,embedding_dim,probs,seed):
    
    hessian = np.zeros(shape=(embedding_dim,embedding_dim))

    for j in range(A.shape[0]):

        if i != j:

            x_i = X[i,:]
            x_j = X[j,:]

            x_diff = (x_i - x_j).reshape(-1,1)  
            
            prob = probs[i,j]#get_pij(i,j,s1,s2,prior,seed=seed)

            h = (gamma**2) * np.dot(x_diff,x_diff.T) * (prob * (1-prob))

            p_diff_mat = gamma * (prob - A[i,j]) * np.identity(h.shape[0])

            hessian += (p_diff_mat - h)
            
    return hessian

def explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A,hessians,probs,seed):

    # hessian = get_hessian(i=i,s1=s1,s2=s2,gamma=gamma,X=X,A=A,
    #     embedding_dim=embedding_dim,prior=prior)

    hessian = hessians[i]

    pij = probs[i,j]#get_pij(i=i,j=j,s1=s1,s2=s2,prior_probs=prior_probs,seed=seed)

    invert = (-hessian) / ((gamma**2 * (pij) * (1-pij)))

    hess_inv = np.linalg.inv(invert)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]
    x_l = X[l,:]

    xij_diff = (x_i - x_j).T

    xlk_diff = (x_l - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xlk_diff).squeeze()

def get_explanations(i,j,s1,s2,embedding_dim,gamma,X,A,top_k,iter_data,hessians,probs,seed):

    temp = []

    for k,l in iter_data:

        if (i,j) != (k,l):

            score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,A,hessians,probs,seed)

            temp.append(((k,l),score))

    sorted_scores = sorted(temp,key=lambda x:x[1], reverse=True)[0:top_k]

    explanation = [np.array(tup) for tup,_ in sorted_scores]

    return np.array(explanation)

if __name__ == '__main__':

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    print(f'CPU count: {joblib.cpu_count()}')

    data = np.load(os.path.join('.','data','royalty_spouse.npz'))

    train = data['X_train']
    test = data['X_test']

    train_exp = data['train_exp']
    test_exp = data['test_exp']

    full_train = np.concatenate((train,train_exp.reshape(-1,3)), axis=0)
    #full_test = np.concatenate((test,test_exp.reshape(-1,3)), axis=0)
    #full_data = np.concatenate((full_train,full_test), axis=0)

    entities = data['entities'].tolist()
    relations = data['relations'].tolist()

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    #idx2ent = {idx:ent for ent,idx in ent2idx.items()}
    #idx2rel = {idx:rel for rel,idx in rel2idx.items()}

    #train2idx = utils.array2idx(full_train,ent2idx,rel2idx)
    train2idx = utils.array2idx(train,ent2idx,rel2idx)
    test2idx = utils.array2idx(test,ent2idx,rel2idx)
    
    trainexp2idx = utils.array2idx(train_exp,ent2idx,rel2idx)
    testexp2idx = utils.array2idx(test_exp,ent2idx,rel2idx)

    adjacency_data = np.concatenate((train,train_exp.reshape(-1,3)), axis=0)
    #adjacency_data = np.concatenate((adjacency_data,test), axis=0)

    A = utils.get_adjacency_matrix(adjacency_data,entities,NUM_ENTITIES)

    trainexp2idx = np.concatenate(
        [trainexp2idx[:,:,0],trainexp2idx[:,:,2]],axis=1).reshape(-1,1,2)

    #testexp2idx = np.concatenate(
        #[testexp2idx[:,:,0],testexp2idx[:,:,2]],axis=1).reshape(-1,1,2)
    testexp2idx = np.concatenate(
        [testexp2idx[:,:,0],testexp2idx[:,:,2]],axis=1).reshape(-1,1,2)

    EMBEDDING_DIM = 50
    S1 = 1
    S2 = 1.5
    LEARNING_RATE = .001
    MAX_ITER = 100
    GAMMA = (1/(S1**2)) - (1/(S2**2))
    TOP_K = 1

    prior = maxent.BGDistr(A) 
    prior.fit()

    CNE = cne.ConditionalNetworkEmbedding(
        A=A,
        d=EMBEDDING_DIM,
        s1=S1,
        s2=S2,
        prior_dist=prior
        )

    CNE.fit(lr=LEARNING_RATE, max_iter=MAX_ITER)

    X = CNE._ConditionalNetworkEmbedding__emb

    probs = joblib.Parallel(n_jobs=-2, verbose=0)(
        joblib.delayed(compute_prob)(
            i,S1,S2,X,A,prior,seed=SEED
            ) for i in range(NUM_ENTITIES)
        )

    PROBS = np.array(probs)

    hessians = joblib.Parallel(n_jobs=-2, verbose=20)(
        joblib.delayed(get_hessian)(
            i,S1,S2,GAMMA,X,A,EMBEDDING_DIM,PROBS,SEED
            ) for i in range(NUM_ENTITIES)
        )

    HESSIANS = np.array(hessians)

    # hessian_file = np.load(os.path.join('.','data','hessians_spouse.npz'))

    # HESSIANS = hessian_file['hessians']
    # PROBS = hessian_file['probs']
    # X = hessian_file['X']
    # S1 = hessian_file['s1']
    # S2 = hessian_file['s2']
    # GAMMA = hessian_file['gamma']
    # LEARNING_RATE = hessian_file['learning_rate']
    # EMBEDDING_DIM = hessian_file['embedding_dim']
    # MAX_ITER = hessian_file['max_iter']
    # TOP_K = 1

    explanations = joblib.Parallel(n_jobs=-2, verbose=0)(
        joblib.delayed(get_explanations)(
            i,j,S1,S2,EMBEDDING_DIM,GAMMA,X,A,TOP_K,testexp2idx.reshape(-1,2),HESSIANS,PROBS,SEED
            ) for i,_,j in test2idx
        )

    explanations = np.array(explanations)

    jaccard = utils.jaccard_score(testexp2idx,explanations)
    #jaccard = utils.jaccard_score(trainexp2idx,explanations)

    print(f"Jaccard score={jaccard} using:")
    print(f"embedding dimensions={EMBEDDING_DIM},s1={S1},s2={S2}")
    print(f"learning_rate={LEARNING_RATE},max_iter={MAX_ITER}")
