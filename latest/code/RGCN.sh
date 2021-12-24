#!/bin/bash

#conda activate kg_env_new

# ./RGCN.py $1 $2 $3 $4 $5
# ./RGCN.py 'royalty_20k' 'spouse' 2000 25 1e-3
# ./RGCN.py 'royalty_20k' 'successor' 1000 25 1e-3
# ./RGCN.py 'royalty_20k' 'predecessor' 2000 25 1e-3
# ./RGCN.py 'royalty_20k' 'full_data' 1000 25 1e-3

# ./RGCN.py 'royalty_30k' 'spouse' 2000 25 1e-3
# ./RGCN.py 'royalty_30k' 'grandparent' 2000 25 1e-3
# ./RGCN.py 'royalty_30k' 'full_data' 1000 25 1e-3

./rgcn_eval.py 'royalty_20k' 'spouse' 25 #accuracy 0.785
./rgcn_eval.py 'royalty_20k' 'successor' 25 #accuracy .692
./rgcn_eval.py 'royalty_20k' 'predecessor' 25 #accuracy .767
./rgcn_eval.py 'royalty_20k' 'full_data' 25 #accuracy .802

./rgcn_eval.py 'royalty_30k' 'spouse' 25 #accuracy 0.785
./rgcn_eval.py 'royalty_30k' 'grandparent' 25 #accuracy 0.714
./rgcn_eval.py 'royalty_30k' 'full_data' 25 #accuracy 0.688