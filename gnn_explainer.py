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

data = np.load(os.path.join('.','data','royalty.npz'))

full_train = data['full_train']
train = data['X_train']
test = data['X_test']

train_exp = data['train_exp']
test_exp = data['test_exp']

entities = data['entities'].tolist()
relations = data['relations'].tolist()

num_entities = len(entities)
num_relations = len(relations)

ent2idx = dict(zip(entities, range(num_entities)))
rel2idx = dict(zip(relations, range(num_relations)))

entity_embeddings = np.load(os.path.join('.','data','transE_embeddings.npz'))['entity_embeddings']

X = torch.tensor([entity_embeddings[ent2idx[h]] for h,_,_ in full_train])
y = torch.tensor([(rel2idx[r]) for _,r,_ in full_train])
ents = np.array([(ent2idx[h],ent2idx[t]) for h,_,t in full_train]).T
ents_flipped = np.stack((ents[1,:], ents[0,:]))
edge_index = torch.tensor(np.concatenate((ents,ents_flipped), axis=1), dtype=torch.long).contiguous()

data = Data(x=X, y=y, edge_index=edge_index,num_classes=num_relations)

X_test = torch.tensor([entity_embeddings[ent2idx[h]] for h,_,_ in test])
y_test = torch.tensor([(rel2idx[r]) for _,r,_ in test])
test_ents = np.array([(ent2idx[h],ent2idx[t]) for h,_,t in test]).T
#test_ents_flipped = np.stack((test_ents[1,:], test_ents[0,:]))
#test_edge_index = torch.tensor(np.concatenate((test_ents,test_ents_flipped), axis=1), dtype=torch.long)
test_edge_index = torch.tensor(test_ents, dtype=torch.long).t().contiguous()
test_data = Data(x_test=X_test,y_test=y_test,test_edge_index=test_edge_index,num_classes=num_relations)

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

x_test,test_y, test_edge_index = test_data.x_test,test_data.y_test,test_data.test_edge_index

def get_explanations(i,x,y,edge_index, explainer):

    node_feat_mask, edge_mask = explainer.explain_node(i, x, edge_index)
    _, G = explainer.visualize_subgraph(i, edge_index, edge_mask, y)

    temp = []
    exp = list(G.edges)
    for tup in exp:
        sorted_tup = tuple(sorted(tup))
        temp.append(sorted_tup)

    return list(set(temp))

# explanations = joblib.Parallel(n_jobs=-2, verbose=20)(
#     joblib.delayed(get_explanations)(i,x_test,test_y,test_edge_index, explainer) for i in range(len(x_test[-2:]),
#         len(x_test))
#     )

explanations = []

for i in range(len(x)):

    exp = get_explanations(i,x,data.y,edge_index, explainer)

    explanations.append(exp)

print(explanations)
#print(utils.jaccard_score(unique_explanations,unique_explanations))
