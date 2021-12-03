# Royalty-20k and Royalty-30k datasets

Loading the data:

```python
import utils
import numpy as np

DATASET = 'royalty_30k'
RULE = 'spouse'

data = np.load(DATASET+'.npz')

triples,traces,entities,relations = utils.get_data(data, RULE)
```

`DATASET` can be either `'royalty_30k' or 'royalty_20k'`

For `Royalty-30k`, `RULE` can be `'spouse'`, `'successor'`, `'predecessor'` or `'full_data'`

For `Royalty-20k`, `RULE` can be `'spouse'`, `'grandparent'` or `'full_data'`

`triples` is a `(N,3)` numpy array containing the triples used for link prediction,  with `N` being the total number of triples  

`traces` is a `(N,M,3)` numpy array containing the explanations for each triple in `triples`, and `M` is either 1 or 2. 

`triples[0]` is a triple we want an explanation for, `traces[0]` gives the 
only explanation for why `triples[0]` is a fact  

`entities` is a numpy array of all unique entities  

`relations` is a numpy array of all unique relations     

## Linked Data Ground Truth for Quantitative and Qualitative Evaluation of Explanations for Relational Graph Convolutional Network Link Prediction on Knowledge Graphs

To reproduce the results from this paper, navigate to the /reproduce_results/ directory, 

then build a conda environment which uses Python 3.7 and Tensorflow-GPU 2.3:
```
conda env create -f kg_env.yml --name kg_env
```

For the most recent benchmark results, navigate to the /latest/ directory

## Please use the following citations: 
```
@inproceedings{halliwell:hal-03430113,
  TITLE = {{Linked Data Ground Truth for Quantitative and Qualitative Evaluation of Explanations for Relational Graph Convolutional Network Link Prediction on Knowledge Graphs}},
  AUTHOR = {Halliwell, Nicholas and Gandon, Fabien and Lecue, Freddy},
  URL = {https://hal.archives-ouvertes.fr/hal-03430113},
  BOOKTITLE = {{International Conference on Web Intelligence and Intelligent Agent Technology}},
  ADDRESS = {Melbourne, Australia},
  YEAR = {2021},
  MONTH = Dec,
  DOI = {10.1145/3486622.3493921},
  KEYWORDS = {link prediction ; Explainable AI ; knowledge graphs ; graph neural networks},
  PDF = {https://hal.archives-ouvertes.fr/hal-03430113v2/file/WI_IAT.pdf},
  HAL_ID = {hal-03430113},
  HAL_VERSION = {v2},
}

@misc{halliwell:hal-03339562,
  TITLE = {{A Simplified Benchmark for Non-ambiguous Explanations of Knowledge Graph Link Prediction using Relational Graph Convolutional Networks}},
  AUTHOR = {Halliwell, Nicholas and Gandon, Fabien and Lecue, Freddy},
  URL = {https://hal.archives-ouvertes.fr/hal-03339562},
  NOTE = {Poster},
  HOWPUBLISHED = {{International Semantic Web Conference}},
  YEAR = {2021},
  MONTH = Oct,
  KEYWORDS = {knowledge graphs ; Explainable AI ; link prediction},
  PDF = {https://hal.archives-ouvertes.fr/hal-03339562v2/file/paper326.pdf},
  HAL_ID = {hal-03339562},
  HAL_VERSION = {v2},
}
```