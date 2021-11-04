#!/bin/bash

#conda activate kg_env_new

./eval.py 'royalty_20k' 'spouse' 25
./eval.py 'royalty_20k' 'successor' 25
./eval.py 'royalty_20k' 'predecessor' 25
./eval.py 'royalty_20k' 'full_data' 25

./eval.py 'royalty_30k' 'spouse' 25
./eval.py 'royalty_30k' 'grandparent' 25
./eval.py 'royalty_30k' 'full_data' 25
