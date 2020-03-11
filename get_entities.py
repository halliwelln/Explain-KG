#!/usr/bin/env python3

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import random as rn
import json
import time

np.random.seed(123)
rn.seed(123)

DATA_PATH = '/data/wimmics/user/nhalliwe/KG-Data/'


fb15k_237 = np.load(DATA_PATH + 'fb15k_237.npz', allow_pickle=True)

entities = list(set(np.concatenate([fb15k_237['train'][:,0], 
                fb15k_237['train'][:,2],
                fb15k_237['valid'][:,0], 
                fb15k_237['valid'][:,2],
                fb15k_237['test'][:,0],
                fb15k_237['test'][:,2]], axis=0)))

print('Number of entities: ', len(entities))

entity_dict = {}
seen = []
missed = []

for str_id in entities:
  
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    query = """
        SELECT ?item ?itemLabel 
        WHERE 
        {
          ?item wdt:P646 '%s' .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    sparql.setQuery(query % str_id)
    
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    try:
        entity_str = results['results']['bindings'][0]['itemLabel']['value']
    except:
        entity_str = None
        missed.append(str_id)
    
    if str_id not in entity_dict:
        
        entity_dict[str_id] = entity_str
        
    else:
        seen.append((str_id, entity_str))
        
    time.sleep(1)

print('Number of entities found: ', len(entity_dict))

print('Dictionary lengths equal: ', len(entities) == len(entity_dict))

with open(DATA_PATH + 'fb15k_237_entities.json', 'w') as json_file:
    
    json.dump(entity_dict, json_file)

np.savez(DATA_PATH + 'missed.npz', a=np.array(seen), b=np.array(missed))
