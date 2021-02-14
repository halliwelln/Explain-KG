#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import utils
import random as rn

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)

data = np.load(os.path.join('..','data','royalty.npz'))

RULE = 'aunt'

triples,traces,nopred,entities,relations = utils.get_data(data,RULE)

i = 7274

plot_triples = np.concatenate([triples[i].reshape(1,3),traces[i].reshape(-1,3)], axis=0)
plot_triples = plot_triples[0:-1,:]

G = nx.MultiDiGraph()
for triple in plot_triples:
    G.add_node(triple[0])
    G.add_node(triple[2])
    G.add_edge(triple[0], triple[2])

pos = nx.spring_layout(G,seed=SEED)

plt.figure(figsize=(7,7))
#plt.figure(figsize=(7,5))

nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='skyblue', alpha=0.9,
        font_size=12,
        labels={node: node for node in G.nodes()})
nx.draw_networkx_edge_labels(G,pos,edge_labels={('Princess_Clémentine_of_Belgium',
                                                 'Archduchess_Hermine_of_Austria'):'hasAunt',
('Marie_Henriette_of_Austria',
 'Archduchess_Hermine_of_Austria'):'hasSister',
('Princess_Clémentine_of_Belgium','Marie_Henriette_of_Austria'):'hasParent'},
                             font_color='black',font_size=12)
ax = plt.gca()
ax.margins(0.25)
plt.axis("off")
plt.savefig('../data/plots/explanation-plot.pdf',bbox_inches='tight')
