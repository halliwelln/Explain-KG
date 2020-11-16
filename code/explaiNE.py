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

def compute_prob(i,s1,s2,X,num_entities,prior,seed):
    prob = []
    for j in range(num_entities):
        prob.append(get_pij(i,j,X,s1,s2,prior,seed))
    return prob

def get_hessian(i,s1,s2,gamma,X,A,embedding_dim,probs,seed):
    
    hessian = np.zeros(shape=(embedding_dim,embedding_dim))

    for j in range(A.shape[0]):

        if i != j:

            x_i = X[i,:]
            x_j = X[j,:]

            x_diff = (x_i - x_j).reshape(-1,1)  
            
            prob = probs[i,j]

            h = (gamma**2) * np.dot(x_diff,x_diff.T) * (prob * (1-prob))

            p_diff_mat = gamma * (prob - A[i,j]) * np.identity(h.shape[0])

            hessian += (p_diff_mat - h)
            
    return hessian

def explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,hessians,probs,seed):

    hessian = hessians[i]

    pij = probs[i,j]

    invert = (-hessian) / ((gamma**2 * (pij) * (1-pij)))

    hess_inv = np.linalg.inv(invert)

    x_i = X[i,:]
    x_j = X[j,:]
    x_k = X[k,:]
    x_l = X[l,:]

    xij_diff = (x_i - x_j).T

    xlk_diff = (x_l - x_k)

    return np.dot(np.dot(xij_diff, hess_inv), xlk_diff).squeeze()

def get_explanations(i,j,s1,s2,embedding_dim,gamma,X,top_k,iter_data,hessians,probs,seed):

    temp = []

    for k,l in iter_data:

        score = explaiNE(i,j,k,l,s1,s2,embedding_dim,gamma,X,hessians,probs,seed)

        temp.append(((k,l),score))

    sorted_scores = sorted(temp,key=lambda x:x[1], reverse=True)[0:top_k]

    explanation = [np.array(tup) for tup,_ in sorted_scores]

    return np.array(explanation)

def jaccard_score(true_exp,pred_exp):

    assert len(true_exp) == len(pred_exp)

    scores = []

    for i in range(len(true_exp)):

        true_i = true_exp[i]
        pred_i = pred_exp[i]

        num_true_traces = true_i.shape[0]
        num_pred_traces = pred_i.shape[0]

        count = 0
        for pred_row in pred_i:
            for true_row in true_i:
                if (pred_row == true_row).all():
                    count +=1

        score = count / (num_true_traces + num_pred_traces-count)

        scores.append(score)
        
    return np.mean(scores)

def get_adjacency_matrix(data,entities,num_entities):

    row = []
    col = []

    for h,r,t in data:

        h_idx = entities.index(h)
        t_idx = entities.index(t)

        row.append(h_idx)
        col.append(t_idx)

    adj = np.ones(len(row))

    return sparse.csr_matrix((adj,(row,col)),shape=(num_entities,num_entities))

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import KFold

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    print(f'CPU count: {joblib.cpu_count()}')

    parser = argparse.ArgumentParser()

    parser.add_argument('rule',type=str,help=
        'Enter which rule to use spouse,successor,...etc (str), -1 (str) for full dataset')
    parser.add_argument('top_k', type=int)
    args = parser.parse_args()

    RULE = args.rule
    TOP_K = args.top_k

    data = np.load(os.path.join('..','data','royalty.npz'))

    if RULE == '-1':
        triples, traces,no_pred_triples,no_pred_traces = utils.concat_triples(data, data['rules'])
        RULE = 'full_data'
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    else:
        triples, traces = data[RULE + '_triples'], data[RULE + '_traces']
        entities = data[RULE + '_entities'].tolist()
        relations = data[RULE + '_relations'].tolist()     

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    EMBEDDING_DIM = 100
    S1 = 1
    S2 = 1.5
    LEARNING_RATE = .001
    MAX_ITER = 100
    GAMMA = (1/(S1**2)) - (1/(S2**2))

    kf = KFold(n_splits=5,shuffle=False,random_state=SEED)

    cv_scores = []
    preds = []

    for train_idx, test_idx in kf.split():

        train = triples[train_idx]
        train = np.concatenate([train,no_pred_triples],axis=0)
        test = triples[test_idx]

        train_exp = traces[train_idx]
        train_exp = np.concatenate([train_exp,no_pred_traces],axis=0)
        test_exp = traces[test_idx]

        train2idx = utils.array2idx(train,ent2idx,rel2idx)
        test2idx = utils.array2idx(test,ent2idx,rel2idx)
        
        trainexp2idx = utils.array2idx(train_exp,ent2idx,rel2idx)
        testexp2idx = utils.array2idx(test_exp,ent2idx,rel2idx)

        adjacency_data = np.concatenate((train,train_exp.reshape(-1,3)), axis=0)

        A = get_adjacency_matrix(adjacency_data,entities,NUM_ENTITIES)

        #trainexp2idx = trainexp2idx[:,:,[0,2]]

        testexp2idx = testexp2idx[:,:,[0,2]]

        prior = maxent.BGDistr(A) 
        prior.fit()

        CNE = cne.ConditionalNetworkEmbedding(
            A=A,
            d=EMBEDDING_DIM,
            s1=S1,
            s2=S2,
            prior_dist=prior
            )

        CNE.fit(lr=LEARNING_RATE,max_iter=MAX_ITER)

        X = CNE._ConditionalNetworkEmbedding__emb

        A = get_adjacency_matrix(test,entities,NUM_ENTITIES)

        PROBS = joblib.Parallel(n_jobs=-2, verbose=0)(
            joblib.delayed(compute_prob)(
                i,S1,S2,X,NUM_ENTITIES,prior,SEED
                ) for i in range(NUM_ENTITIES)
            )

        PROBS = np.array(PROBS)

        HESSIANS = joblib.Parallel(n_jobs=-2, verbose=20)(
            joblib.delayed(get_hessian)(
                i,S1,S2,GAMMA,X,A,EMBEDDING_DIM,PROBS,SEED
                ) for i in range(NUM_ENTITIES)
            )

        HESSIANS = np.array(HESSIANS)
        ITER_DATA = np.unique(testexp2idx.reshape(-1,2), axis=0)

        explanations = joblib.Parallel(n_jobs=-2, verbose=0)(
            joblib.delayed(get_explanations)(
                i,j,S1,S2,EMBEDDING_DIM,GAMMA,X,TOP_K,ITER_DATA,HESSIANS,PROBS,SEED
                ) for i,_,j in test2idx
            )

        explanations = np.array(explanations)

        jaccard = jaccard_score(testexp2idx,explanations)

        cv_scores.append(jaccard)
        preds.append(explanations)

    best_idx = np.argmin(cv_scores)
    best_preds = preds[best_idx]

    # np.savez(os.path.join('.','data','explaine_',RULE,'_preds','.npz'),
    #     preds=best_preds,embedding_dim=EMBEDDING_DIM,learning_rate=LEARNING_RATE,
    #     max_iter=MAX_ITER,s1=S1,s2=S2
    #     )

    print(f"{RULE} jaccard score={np.mean(cv_scores)} using:")
    print(f"embedding dimensions={EMBEDDING_DIM},s1={S1},s2={S2}")
    print(f"learning_rate={LEARNING_RATE},max_iter={MAX_ITER}")
