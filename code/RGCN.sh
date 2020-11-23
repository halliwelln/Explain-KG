#!/bin/bash

conda activate kg_env

./RGCN.py 'spouse' 50
#./RGCN.py 'uncle' 2000
#./RGCN.py 'aunt' 2000 
#./RGCN.py 'successor' 2000
#./RGCN.py 'predecessor' 2000 
#./RGCN.py 'grandparent' 2000 
#./RGCN.py 'full_data' 50