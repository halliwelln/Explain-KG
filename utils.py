#!/usr/bin/env python3

import numpy as np

def get_data(data,rule):

    if rule == 'full_data':

        triples,traces = concat_triples(data, data['rules'])
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples,traces = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,entities,relations

def concat_triples(data, rules):

    triples = []
    traces = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'

        triples.append(data[triple_name])
        traces.append(data[traces_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    
    return triples, traces

