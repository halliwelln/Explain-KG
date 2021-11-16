# Linked Data Ground Truth for Quantitative and Qualitative Evaluation of Explanations for Relational Graph Convolutional Network Link Prediction on Knowledge Graphs

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
  PDF = {https://hal.archives-ouvertes.fr/hal-03430113/file/WI_IAT_2021.pdf},
  HAL_ID = {hal-03430113},
  HAL_VERSION = {v1},
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