#!/bin/bash

#conda activate kg_env_new

#./explaiNE.py $1 $2 $3 $4
# ./explaiNE.py 'royalty_20k' 'spouse' 10
# ./explaiNE.py 'royalty_20k' 'successor' 10
# ./explaiNE.py 'royalty_20k' 'predecessor' 10
# ./explaiNE.py 'royalty_20k' 'full_data' 10

# ./explaiNE.py 'royalty_30k' 'spouse' 10
# ./explaiNE.py 'royalty_30k' 'grandparent' 10
#./explaiNE.py 'royalty_30k' 'full_data' 10

# ./eval.py 'royalty_20k' 'spouse' 10
#./eval.py 'royalty_20k' 'successor' 10
#./eval.py 'royalty_20k' 'predecessor' 10
./eval.py 'royalty_20k' 'full_data' 10

#./eval.py 'royalty_30k' 'spouse' 10
# ./eval.py 'royalty_30k' 'grandparent' 10
./eval.py 'royalty_30k' 'full_data' 10