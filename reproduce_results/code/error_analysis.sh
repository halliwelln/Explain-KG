#!/bin/bash

conda activate kg_env_new

./error_analysis.py 'royalty_20k' 'spouse' 1 'explaine'
./error_analysis.py 'royalty_20k' 'successor' 1 'explaine'
./error_analysis.py 'royalty_20k' 'predecessor' 1 'explaine'

./error_analysis.py 'royalty_30k' 'spouse' 1 'explaine'
./error_analysis.py 'royalty_30k' 'grandparent' 2 'explaine'

./error_analysis.py 'royalty_20k' 'spouse' 1 'gnn_explainer'
./error_analysis.py 'royalty_20k' 'successor' 1 'gnn_explainer'
./error_analysis.py 'royalty_20k' 'predecessor' 1 'gnn_explainer'

./error_analysis.py 'royalty_30k' 'spouse' 1 'gnn_explainer'
./error_analysis.py 'royalty_30k' 'grandparent' 2 'gnn_explainer'