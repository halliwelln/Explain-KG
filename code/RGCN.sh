#!/bin/bash

conda activate kg_env_new

./RGCN.py $1 $2
#./RGCN.py 'spouse' 50
#./RGCN.py 'uncle' 200
#./RGCN.py 'aunt' 200
#./RGCN.py 'successor' 200
#./RGCN.py 'predecessor' 200
#./RGCN.py 'grandparent' 200
#./RGCN.py 'full_data' 50