#!/bin/bash

conda activate kg_env_new

./explaiNE.py $1 $2 $3 $4
#./explaiNE.py 'spouse' 1
#./explaiNE.py 'uncle' 2
#./explaiNE.py 'aunt' 2
#./explaiNE.py 'successor' 1
#./explaiNE.py 'predecessor' 1
#./explaiNE.py 'grandparent' 2
#./explaiNE.py 'full_data' 2