#!/usr/bin/env python3
    
import os
import utils
import random as rn
import argparse
import numpy as np

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str,
    help='royalty_30k or royalty_20k')
parser.add_argument('rule',type=str,
    help='spouse,successor,...,full_data')
parser.add_argument('embedding_dim',type=int)
parser.add_argument('trace_length',type=int)

args = parser.parse_args()

DATASET = args.dataset
RULE = args.rule
TRACE_LENGTH = args.trace_length

data = np.load(os.path.join('..','data',DATASET+'.npz'))

triples,traces,entities,relations = utils.get_data(data,RULE)
###################################################

#if RULE != 'full_data':
gnn_data = np.load(
    os.path.join('..','data','preds',DATASET,
        'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

gnn_true_exps = traces[gnn_data['test_idx']]

gnn_preds = gnn_data['preds']

num_gnn_triples = gnn_true_exps.shape[0]
gnn_jaccard = 0.0
for i in range(num_gnn_triples):
    gnn_jaccard += utils.jaccard_score(gnn_true_exps[i],gnn_preds[i])
gnn_jaccard /= num_gnn_triples

gnn_precision, gnn_recall = utils.precision_recall(gnn_true_exps,gnn_preds)
gnn_f1 = utils.f1(gnn_precision,gnn_recall)

print(f'{DATASET} {RULE} GnnExplainer')
print(f'precision {round(gnn_precision,3)}')
print(f'recall {round(gnn_recall,3)}')
print(f'f1 {round(gnn_f1,3)}')
print(f'jaccard score: {round(gnn_jaccard,3)}')

###################################################

explaine_data = np.load(
    os.path.join('..','data','preds',DATASET,
        'explaine_'+DATASET+'_'+RULE+'_preds.npz'),allow_pickle=True)

explaine_true_exps = traces[explaine_data['test_idx']]

explaine_preds = explaine_data['preds']

num_explaine_triples = explaine_true_exps.shape[0]
explaine_jaccard = 0.0
for i in range(num_explaine_triples):
    explaine_jaccard += utils.jaccard_score(explaine_true_exps[i],explaine_preds[i])
explaine_jaccard /= num_explaine_triples

explaine_precision, explaine_recall = utils.precision_recall(explaine_true_exps,explaine_preds)
explaine_f1 = utils.f1(explaine_precision,explaine_recall)

print(f'{DATASET} {RULE} ExplaiNE')
print(f'precision {round(explaine_precision,3)}')
print(f'recall {round(explaine_recall,3)}')
print(f'f1 {round(explaine_f1,3)}')
print(f'jaccard score: {round(explaine_jaccard,3)}')

# NUM_ENTITIES = len(entities)
# NUM_RELATIONS = len(relations)

# ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
# rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

# if RULE != 'full_data':

#     gnn_data = np.load(os.path.join('..','data','preds','gnn_explainer_'+RULE+'_preds.npz'),allow_pickle=True)

#     gnn_exp2idx = utils.array2idx(traces[gnn_data['test_idx']],ent2idx,rel2idx)
#     gnn_num_triples = gnn_exp2idx.shape[0]
    
#     all_gnn_preds = gnn_data['preds']
    
#     gnn_true_exps = get_true_exps(gnn_exp2idx,gnn_num_triples, TRACE_LENGTH)

#     gnn_preds = []
#     for i in range(all_gnn_preds.shape[0]):
#         preds_i = []
#         for idx, j in enumerate(all_gnn_preds[i]):
#             if j.shape[0] > 0:
#                 rel = np.ones((j.shape[0]),dtype=np.int64) * idx
#                 preds_i.append(np.column_stack((j[:,0],rel,j[:,1])))            
#         gnn_preds.append(np.concatenate(preds_i, axis=0))

#     gnn_precision, gnn_recall = eval(gnn_true_exps,gnn_preds,gnn_num_triples)
#     print(f"GnnExplainer precision {gnn_precision}, GnnExplainer recall {gnn_recall}")

# explaine_data = np.load(os.path.join('..','data','preds','explaine_'+RULE+'_preds.npz'),allow_pickle=True)

# explaine_exp2idx = utils.array2idx(traces[explaine_data['test_idx']],ent2idx,rel2idx)
# explaine_num_triples = explaine_exp2idx.shape[0]

# explaine_true_exps = get_true_exps(explaine_exp2idx,explaine_num_triples, TRACE_LENGTH)

# explaine_precision, explaine_recall = eval(explaine_true_exps,explaine_data['preds'],explaine_num_triples)
# print(f"explaiNE precision {explaine_precision}, explaiNE recall {explaine_recall}")

