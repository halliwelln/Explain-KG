#!/usr/bin/env python3

import numpy as np
import random as rn
import os
from scipy.stats import halfnorm
import utils
import joblib
from scipy import sparse

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

def jaccard_score(true_exp,pred_exp,top_k,return_scores=False):

    assert len(true_exp) == len(pred_exp)

    scores = []

    for i in range(len(true_exp)):

        true_i = true_exp[i][:top_k,]
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
        
    if return_scores:
        return np.mean(scores), np.array(scores)
    else:
        return np.mean(scores)

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import KFold
    import maxent

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    print(f'CPU count: {joblib.cpu_count()}')

    parser = argparse.ArgumentParser()

    parser.add_argument('rule',type=str,help=
        'Enter which rule to use spouse,successor,...etc (str), full_data for full dataset')
    parser.add_argument('top_k', type=int)
    args = parser.parse_args()

    RULE = args.rule
    TOP_K = args.top_k

    data = np.load(os.path.join('..','data','royalty.npz'))

    if RULE == 'full_data':
        triples,traces,nopred = utils.concat_triples(data, data['rules'])
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    else:
        triples,traces,nopred = utils.concat_triples(data, [RULE,'brother','sister'])
        sister_relations = data['sister_relations'].tolist()
        sister_entities = data['sister_entities'].tolist()

        brother_relations = data['brother_relations'].tolist()
        brother_entities = data['brother_entities'].tolist()

        entities = np.unique(data[RULE + '_entities'].tolist()+brother_entities+sister_entities).tolist()
        relations = np.unique(data[RULE + '_relations'].tolist()+brother_relations+sister_relations).tolist()

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    cne_data = np.load(os.path.join('..','data','weights','cne_embeddings_'+RULE+'.npz'))

    X = cne_data['embeddings']
    S1 = cne_data['s1']
    S2 = cne_data['s2']
    EMBEDDING_DIM = X.shape[1]
    GAMMA = (1/(S1**2)) - (1/(S2**2))

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

    cv_scores = []
    preds = []

    for train_idx, test_idx in kf.split(X=triples):

        #test_idx = test_idx[0:10]

        train2idx = utils.array2idx(triples[train_idx],ent2idx,rel2idx)
        trainexp2idx = utils.array2idx(traces[train_idx],ent2idx,rel2idx)
        nopred2idx = utils.array2idx(nopred,ent2idx,rel2idx)

        adjacency_data = np.concatenate([train2idx,trainexp2idx.reshape(-1,3),nopred2idx],axis=0)

        test2idx = utils.array2idx(triples[test_idx],ent2idx,rel2idx)
        testexp2idx = utils.array2idx(traces[test_idx],ent2idx,rel2idx)

        A = utils.get_adjacency_matrix(np.unique(adjacency_data,axis=0),NUM_ENTITIES)

        prior = maxent.BGDistr(A) 
        prior.fit()

        #trainexp2idx = trainexp2idx[:,:,[0,2]]

        testexp2idx = testexp2idx[:,:,[0,2]]
        
        A = utils.get_adjacency_matrix(test2idx,NUM_ENTITIES)

        PROBS = joblib.Parallel(n_jobs=-2, verbose=20)(
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

        ITER_DATA = np.concatenate([test2idx,np.unique(testexp2idx.reshape(-1,2), axis=0)],axis=0)#add test2idx

        explanations = joblib.Parallel(n_jobs=-2, verbose=20)(
            joblib.delayed(get_explanations)(
                i,j,S1,S2,EMBEDDING_DIM,GAMMA,X,TOP_K,ITER_DATA,HESSIANS,PROBS,SEED
                ) for i,_,j in test2idx
            )

        explanations = np.array(explanations)

        jaccard = jaccard_score(testexp2idx,explanations,TOP_K)

        cv_scores.append(jaccard)
        preds.append(explanations)

    best_idx = np.argmin(cv_scores)
    best_preds = preds[best_idx]

    # np.savez(os.path.join('..','data','preds','explaine_'+RULE+'_preds.npz'),
    #     preds=best_preds,embedding_dim=EMBEDDING_DIM,s1=S1,s2=S2,best_idx=best_idx
    #     )

    print(f"{RULE} jaccard score={np.mean(cv_scores)} using:")
    print(f"embedding dimensions={EMBEDDING_DIM},s1={S1},s2={S2}")
