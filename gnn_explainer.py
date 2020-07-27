#!/usr/bin/env python3

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GNNExplainer
from torch.nn import Sequential, Linear
import numpy as np
import os
import random as rn
from torch_geometric.data import Data
import joblib
import utils

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
rn.seed(SEED)
torch.manual_seed(SEED)

d = np.array([('Sri_Vikrama_Rajasinha_of_Kandy','successor','British_Ceylon'),
('William_IV_of_the_United_Kingdom','predecessor','George_IV_of_the_United_Kingdom'),
('Tvrtko_II_of_Bosnia','successor','Ostoja_of_Bosnia'),
('Alexis_of_Russia','spouse','Maria_Ilyinichna_Miloslavskaya'),
('Haakon_VII_of_Norway','successor','Olav_V_of_Norway')])

entities = list(np.unique(np.concatenate((d[:,0],d[:,2]))))
relations = list(np.unique(d[:,1]))

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

X = torch.randn((num_entities, 2))
y = torch.randint(num_relations, (num_entities,))
a = np.array([(ent2idx[h],ent2idx[t]) for h,_,t in d]).T
b = np.stack((a[1,:], a[0,:]))
edge_index = torch.tensor(np.concatenate((a,b), axis=1), dtype=torch.long)

data = Data(x=X, y=y, edge_index=edge_index,num_classes=num_relations)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin = Sequential(Linear(10,10))
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, edge_index = data.x, data.edge_index

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits, data.y)
    loss.backward()
    optimizer.step()

explainer = GNNExplainer(model, epochs=200)

def get_explanations(i,x,y,edge_index, explainer):

    node_feat_mask, edge_mask = explainer.explain_node(i, x, edge_index)
    _, G = explainer.visualize_subgraph(i, edge_index, edge_mask, y=y)

    return list(G.edges)

explanations = joblib.Parallel(n_jobs=-2, verbose=20)(
    joblib.delayed(get_explanations)(i,x,data.y,edge_index, explainer) for i in range(2)
    )

def get_unique_explanations(i, explanations):

    temp = []

    for tup in explanations[i]:
        sorted_tup = tuple(sorted(tup))
        temp.append(sorted_tup)

    return list(set(temp))

unique_explanations = joblib.Parallel(n_jobs=-2,verbose=20)(
    joblib.delayed(get_unique_explanations)(i, explanations) for i in range(len(explanations))
    )

print(utils.jaccard_score(unique_explanations,unique_explanations))