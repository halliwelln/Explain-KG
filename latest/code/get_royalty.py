#!/usr/bin/env python3

from SPARQLWrapper import SPARQLWrapper, XML
from rdflib import Graph
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('start',type=int)
parser.add_argument('offset', type=int)
args = parser.parse_args()

start = args.start #0
offset = args.offset #1000
end = 20000
limit = 1000

failed = []

for i in range(start,end+offset,offset):
    try:
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery(f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            CONSTRUCT{{
            ?x
            dbo:parent ?parent ;
            dbo:predecessor ?predecessor ;
            dbo:spouse ?spouse ;
            foaf:gender ?gender ;
            dbo:successor ?sucessor .
            }}
            WHERE {{
            ?x a dbo:Royalty .
            OPTIONAL {{?x dbo:parent ?parent }}
            OPTIONAL {{?x dbo:predecessor ?predecessor }}
            OPTIONAL {{?x dbo:spouse ?spouse }}
            OPTIONAL {{?x foaf:gender ?gender }}
            OPTIONAL {{?x dbo:successor ?sucessor }}
            }}
            Order by asc(?x) limit {limit} offset {i}
            """)
        sparql.setReturnFormat(XML)
        results = sparql.query().convert().serialize(
            destination=os.path.join('..','data','royalty_temp',f'sparql-{i}'),
            format='xml'
            )
        
    except:
        failed.append(i)

if failed:
    print(failed)

g = Graph()

for i in range(start,end+offset,offset):

    g.parse(os.path.join('..','data','royalty_temp',f'sparql-{i}'), format="xml")

g.serialize(destination=os.path.join('..','data','rules','full_royalty'),format='xml')