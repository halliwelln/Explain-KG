#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import utils
import random as rn
import tensorflow as tf

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)

data = np.load(os.path.join('..','data','royalty.npz'))

RULE = 'aunt'
TRACE_LENGTH = 2

triples,traces,nopred,entities,relations = utils.get_data(data,RULE)

NUM_ENTITIES = len(entities)
NUM_RELATIONS = len(relations)

ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

idx2ent = dict(zip(range(NUM_ENTITIES),entities))
idx2rel = dict(zip(range(NUM_RELATIONS),relations))

i = 7274

adj_data = np.concatenate([triples,traces[:,0:TRACE_LENGTH,:].reshape(-1,3)],axis=0)

adj_data_sparse = utils.array2idx(adj_data,ent2idx,rel2idx)

adj_mats = utils.get_adj_mats(
    data=adj_data_sparse,
    num_entities=NUM_ENTITIES,
    num_relations=NUM_RELATIONS
)

head = triples[i][0]
tail = triples[i][2]

head_idx = ent2idx[head]
tail_idx = ent2idx[tail]

neighbor_indices = []

for rel_idx in range(NUM_RELATIONS):

    dense_mat = tf.sparse.to_dense(adj_mats[rel_idx]).numpy()[0]

    head_neighbors = np.argwhere(dense_mat[head_idx,:]).flatten()
    tail_neighbors = np.argwhere(dense_mat[:,tail_idx]).flatten()
    
    head_triples = [(head_idx,rel_idx,t_idx) for t_idx in head_neighbors]
    tail_triples = [(h_idx,rel_idx,tail_idx) for h_idx in tail_neighbors]
    
    if head_triples:
        neighbor_indices.append(head_triples)
    if tail_triples:
        neighbor_indices.append(tail_triples)

plot_triples = utils.idx2array(np.concatenate(neighbor_indices,axis=0),idx2ent,idx2rel)[5:]

label_dict = {}

for trip in plot_triples:
    
    head,rel,tail = trip
    
    label_dict[(head,tail)] = 'has' + rel.title()

plt.figure(figsize=(10,10))

G = nx.MultiDiGraph()
for triple in plot_triples:
    G.add_node(triple[0])
    G.add_node(triple[2])
    G.add_edge(triple[0], triple[2])
    
ground_truth = np.concatenate([triples[i].reshape(-1,3),traces[i]],axis=0)[0:-1]

gt_entities = np.unique(np.concatenate([ground_truth[:,0],ground_truth[:,2]]))

node_sizes = [1200 if ent in gt_entities else 300 for ent in G.nodes() ]

pos = nx.spring_layout(G,seed=SEED,k=10)

nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
        node_size=node_sizes, node_color='skyblue', alpha=0.9,font_size=12,
        labels={node: node for node in G.nodes()})

nx.draw_networkx_edge_labels(G,pos,edge_labels=label_dict,font_color='black',font_size=10)

ax = plt.gca()
ax.margins(0.25)
plt.axis("off")
plt.savefig('../data/plots/explanation-plot.pdf',bbox_inches='tight')

# i = 7274

# plot_triples = np.concatenate([triples[i].reshape(1,3),traces[i].reshape(-1,3)], axis=0)
# plot_triples = plot_triples[0:-1,:]

# G = nx.MultiDiGraph()
# for triple in plot_triples:
#     G.add_node(triple[0])
#     G.add_node(triple[2])
#     G.add_edge(triple[0], triple[2])

# pos = nx.spring_layout(G,seed=SEED)

# plt.figure(figsize=(7,7))
# #plt.figure(figsize=(7,5))

# nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
#         node_size=500, node_color='skyblue', alpha=0.9,
#         font_size=12,
#         labels={node: node for node in G.nodes()})
# nx.draw_networkx_edge_labels(G,pos,edge_labels={('Princess_Clémentine_of_Belgium',
#                                                  'Archduchess_Hermine_of_Austria'):'hasAunt',
# ('Marie_Henriette_of_Austria',
#  'Archduchess_Hermine_of_Austria'):'hasSister',
# ('Princess_Clémentine_of_Belgium','Marie_Henriette_of_Austria'):'hasParent'},
#                              font_color='black',font_size=12)
# ax = plt.gca()
# ax.margins(0.25)
# plt.axis("off")
# plt.savefig('../data/plots/explanation-plot.pdf',bbox_inches='tight')
